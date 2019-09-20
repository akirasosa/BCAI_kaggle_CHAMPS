import dataclasses
import logging
import os
import pickle
import shutil
from multiprocessing import cpu_count
from pathlib import Path
from pprint import pformat
from time import time
from typing import Dict, Callable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch.nn.utils import weight_norm
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torch_scatter import scatter_add
from torch_scatter import scatter_mean

from my_lib.torch.modules import MLP
from my_lib.common.avg_meter import AverageMeterSet
from my_lib.common.early_stopping import EarlyStopping
from my_lib.torch.funcs import sqdist, batched_index_select
from my_lib.torch.optim import RAdam
from proj import const
from proj.loader import PandasDataset, atoms_collate_fn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.display.width = 999


# %%
@dataclasses.dataclass
class Conf:
    lr: float = 1e-4
    weight_decay: float = 1e-4

    clr_max_lr: float = 3e-3
    clr_base_lr: float = 3e-6
    clr_gamma: float = 0.999991

    train_batch: int = 32
    val_batch: int = 256

    tformer_dim: int = 650
    tformer_n_layers: int = 14
    tformer_d_inner: int = 3800
    tformer_dropout: float = 0.03
    tformer_dropatt: float = 0.03
    tformer_n_head: int = 10
    tformer_wnorm: bool = True

    optim: str = 'adam'
    # loss: str = 'g_log_mae'

    epochs: int = 400
    is_save_epoch_fn: Callable = None
    resume_from: Dict[str, int] = None

    db_path: str = None

    seed: int = 1

    is_one_cv: bool = True

    device: str = device

    exp_name: str = 'simple_tformer'
    exp_time: float = time()

    logger_epoch = None

    @staticmethod
    def create_logger(name, filename):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not logger.hasHandlers():
            logger.addHandler(logging.FileHandler(filename))
        return logger

    def __post_init__(self):
        if self.resume_from is not None:
            assert self.out_dir.exists(), f'{self.out_dir} does not exist.'

        self.out_dir.mkdir(exist_ok=True)
        self.logger_epoch = self.create_logger(f'epoch_logger_{self.exp_time}', self.out_dir / 'epoch.log')

        with (self.out_dir / 'conf.txt').open('w') as f:
            f.write(str(self))

        shutil.copy(os.path.realpath(__file__), str(self.out_dir))

        global device
        device = self.device

    @property
    def out_dir(self) -> Path:
        return const.DATA_DIR / 'experiments' / self.exp_name / self.exp_time

    def __str__(self):
        return pformat(dataclasses.asdict(self))


@dataclasses.dataclass
class AtomsData:
    atom_type: torch.Tensor
    atom_pos: torch.Tensor
    scc_idx: torch.Tensor
    scc_type: torch.Tensor
    scc_val: torch.Tensor
    scc_scaler: torch.Tensor


class GraphLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_head, dropout=0.0, attn_dropout=0.0, wnorm=False, lev=0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.lev = lev

        # To produce the query-key-value for the self-attention computation
        self.qkv_net = nn.Linear(d_model, 3 * d_model)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.gamma = nn.Parameter(torch.ones(1))  # For different sub-matrices of D
        self.sqrtd = np.sqrt(d_head)

        if wnorm:
            self.wnorm()

    def wnorm(self):
        self.qkv_net = weight_norm(self.qkv_net, name="weight")
        self.o_net = weight_norm(self.o_net, name="weight")
        self.proj1 = weight_norm(self.proj1, name="weight")
        self.proj2 = weight_norm(self.proj2, name="weight")

    def forward(self, Z, D, new_mask, mask, store=False):
        bsz, n_elem, nhid = Z.size()
        n_head, d_head, d_model = self.n_head, self.d_head, self.d_model
        assert nhid == d_model, "Hidden dimension of Z does not agree with d_model"

        # Self-attention
        inp = Z
        Z = self.norm1(Z)
        Z2, Z3, Z4 = self.qkv_net(Z).view(bsz, n_elem, n_head, 3 * d_head).chunk(3, dim=3)  # "V, Q, K"
        W = torch.einsum('bnij, bmij->binm', Z3, Z4).type(D.dtype) / self.sqrtd
        W = W + new_mask[:, None] - (self.gamma * D)[:, None]
        W = self.attn_dropout(F.softmax(W, dim=3).type(mask.dtype) * mask[:, None])  # softmax(-gamma*D + Q^TK)
        if store:
            pickle.dump(W.cpu().detach().numpy(), open(f'analysis/layer_{self.lev}_W.pkl', 'wb'))
        attn_out = torch.einsum('binm,bmij->bnij', W, Z2.type(W.dtype)).contiguous().view(bsz, n_elem, d_model)
        attn_out = self.dropout(self.o_net(F.leaky_relu(attn_out)))
        Z = attn_out + inp

        # Position-wise feed-forward
        inp = Z
        Z = self.norm2(Z)

        return self.proj2(self.dropout(F.relu(self.proj1(Z)))) + inp


class GraphTransformer(nn.Module):
    def __init__(self, dim, n_layers, d_inner,
                 dropout=0.0,
                 dropatt=0.0,
                 n_head=10,
                 wnorm=False):
        super().__init__()

        self.atom_embedding = nn.Embedding(const.N_ATOMS + 1, dim, padding_idx=0)
        self.layers = nn.ModuleList([
            GraphLayer(
                d_model=dim,
                d_inner=d_inner,
                n_head=n_head,
                d_head=dim // n_head,
                dropout=dropout,
                attn_dropout=dropatt,
                wnorm=wnorm,
                lev=i + 1,
            )
            for i in range(n_layers)
        ])
        self.pair_mlp = MLP(n_in=dim * 2, n_out=1, n_layers=2)
        self.apply(self.weights_init)

    def forward(self, inputs: AtomsData):
        D = sqdist(inputs.atom_pos[:, :, :3],
                   inputs.atom_pos[:, :, :3]).to(device)

        mask = inputs.atom_type[:, :, 0] > 0
        mask = torch.einsum('bi, bj->bij', mask, mask).type(inputs.atom_pos.dtype)

        new_mask = -1e20 * torch.ones_like(mask).to(mask.device)
        new_mask[mask > 0] = 0

        Z = self.atom_embedding(inputs.atom_type[:, :, 0])

        for i in range(len(self.layers)):
            Z = self.layers[i](Z, D, new_mask, mask, store=False)

        x_idx_0 = batched_index_select(Z, 1, inputs.scc_idx[:, :, 0])
        x_idx_1 = batched_index_select(Z, 1, inputs.scc_idx[:, :, 1])
        x_pair = torch.cat((x_idx_0, x_idx_1), dim=2)
        y_pred = self.pair_mlp(x_pair)

        return y_pred

    @staticmethod
    def init_weight(weight):
        nn.init.uniform_(weight, -0.1, 0.1)

    @staticmethod
    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1 or classname.find('Conv1d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                GraphTransformer.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                GraphTransformer.init_bias(m.bias)


def calc_grouped_mae(y_pred, y_true, y_types, y_scaler):
    y_pred_scaled = y_pred.squeeze(dim=2) * y_scaler[:, :, 1] + y_scaler[:, :, 0]
    abs_err = (y_pred_scaled - y_true.squeeze(dim=2)).abs()
    mae_types = scatter_mean(abs_err.view(-1), y_types.view(-1))[1:]  # 0 is pad
    cnt_types = scatter_add(torch.ones_like(abs_err.view(-1)), y_types.view(-1))[1:]

    return mae_types, cnt_types


def run_on_step(batch, meters, model):
    inputs = AtomsData(**{
        k: v.to(device)
        for k, v in batch.items()
    })
    y_pred = model(inputs)

    mae_types, cnt_types = calc_grouped_mae(y_pred, inputs.scc_val, inputs.scc_type, inputs.scc_scaler)

    # loss
    nonzero_indices = cnt_types.nonzero()
    loss = torch.log(mae_types[nonzero_indices] + 1e-9).mean()
    n_pairs = cnt_types.sum()
    meters.update('loss', loss.item(), n_pairs.item())

    for n, (mae, cnt) in enumerate(zip(mae_types, cnt_types)):
        meters.update(f'mae_{const.TYPES[n]}', mae.item(), cnt.item())

    return loss


def run_after_step(meters):
    # log mae for each types
    lmae_types = {
        f'lmae_{t}': np.log(meters[t].avg)
        for t in const.TYPES
    }

    # competition metric
    mean_lmae = np.log([meters[t].avg for t in const.TYPES]).mean()

    return {
        **lmae_types,
        'mean_lmae': mean_lmae,
    }


def train(loader, model: nn.Module, optimizer: Optimizer, scheduler, conf: Conf, prefix: str = 'train'):
    meters = AverageMeterSet()
    model.train()

    for step, batch in enumerate(loader):
        meters.update('lr', optimizer.param_groups[0]['lr'])

        loss = run_on_step(batch, meters, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    metrics = run_after_step(meters)

    return {
        'lr': meters['lr'].avg,
        f'{prefix}_loss': meters['loss'].avg,
        **{
            f'{prefix}_{k}': v
            for k, v in metrics
        }
    }


def validate(loader, model: nn.Module, conf: Conf, prefix: str = 'val'):
    meters = AverageMeterSet()
    model.eval()

    for step, batch in enumerate(loader):
        with torch.no_grad():
            run_on_step(batch, meters, model)

    metrics = run_after_step(meters)

    return {
        f'{prefix}_loss': meters['loss'].avg,
        **{
            f'{prefix}_{k}': v
            for k, v in metrics
        }
    }


def log_hist(df_hist: pd.DataFrame, logger: logging.Logger):
    last = df_hist.tail(1)
    best = df_hist.sort_values('val_mean_lmae', ascending=True).head(1)
    summary = pd.concat((last, best)).reset_index(drop=True)
    summary['name'] = ['Last', 'Best']
    logger.debug(summary[[
                             'name',
                             'epoch',
                             'train_loss',
                             'val_loss',
                             'train_mean_lmae',
                             'val_mean_lmae',
                         ] + [
                             f'train_lmae_{t}' for t in const.TYPES
                         ] + [
                             f'val_lmae_{t}' for t in const.TYPES
                         ]])
    logger.debug('')


def write_on_board(df_hist: pd.DataFrame, writer: SummaryWriter, conf: Conf):
    row = df_hist.tail(1).iloc[0]

    writer.add_scalars(f'{conf.exp_name}/lr', {
        f'{conf.exp_time}': row.lr,
    }, row.epoch)

    writer.add_scalars(f'{conf.exp_name}/loss/coupling/total', {
        f'{conf.exp_time}_train': row.train_loss,
        f'{conf.exp_time}_val': row.val_loss,
    }, row.epoch)

    for tag in conf.types:
        writer.add_scalars(f'{conf.exp_name}/metric/type/{tag}', {
            f'{conf.exp_time}_train': row[f'train_lmae_{tag}'],
            f'{conf.exp_time}_val': row[f'val_lmae_{tag}'],
        }, row.epoch)
    writer.add_scalars(f'{conf.exp_name}/metric/type/total', {
        f'{conf.exp_time}_train': row['train_mean_lmae'],
        f'{conf.exp_time}_val': row['val_mean_lmae'],
    }, row.epoch)


def main(conf: Conf):
    print(conf)
    print(f'less +F {conf.out_dir}/epoch.log')

    df = pd.read_pickle(conf.db_path)
    df = df[~df.is_test]

    folds = KFold(n_splits=4, random_state=conf.seed, shuffle=True)

    for cv, (train_idx, val_idx) in enumerate(folds.split(df)):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        print(cv, len(df_train), len(df_val))

        train_loader = DataLoader(PandasDataset(df_train),
                                  batch_size=conf.train_batch,
                                  shuffle=True,
                                  num_workers=cpu_count() - 1,
                                  collate_fn=atoms_collate_fn)
        val_loader = DataLoader(PandasDataset(df_val),
                                batch_size=conf.val_batch,
                                shuffle=False,
                                num_workers=cpu_count() - 1,
                                collate_fn=atoms_collate_fn)

        model = GraphTransformer(
            dim=conf.tformer_dim,
            n_layers=conf.tformer_n_layers,
            d_inner=conf.tformer_d_inner,
            dropout=conf.tformer_dropatt,
            dropatt=conf.tformer_dropatt,
            n_head=conf.tformer_n_head,
            wnorm=conf.tformer_wnorm,
        ).to(device)

        if conf.optim == 'adam':
            opt = Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        elif conf.optim == 'radam':
            opt = RAdam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        else:
            raise Exception(f'Not supported optim {conf.optim}')
        scheduler = CyclicLR(
            opt,
            base_lr=conf.clr_base_lr,
            max_lr=conf.clr_max_lr,
            step_size_up=len(train_loader) * 10,
            mode="exp_range",
            gamma=conf.clr_gamma,
            cycle_momentum=False,
        )
        early_stopping = EarlyStopping(patience=100)

        if conf.resume_from is not None:
            cv_resume = conf.resume_from['cv']
            start_epoch = conf.resume_from['epoch']
            if cv < cv_resume:
                continue
            ckpt = torch.load(f'{conf.out_dir}/{cv}-{start_epoch:03d}.ckpt')
            model.load_state_dict(ckpt['model'])
            opt.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            writer = SummaryWriter(logdir=ckpt['writer_logdir'], purge_step=start_epoch)
            hist = pd.read_csv(f'{conf.out_dir}/{cv}.csv').to_dict('records')
            print(f'Loaded checkpoint cv {cv}, epoch {start_epoch} from {conf.out_dir}')
        else:
            hist = []
            writer = SummaryWriter(logdir=str(conf.out_dir / 'tb_log'))
            start_epoch = 0

        for epoch in range(start_epoch, conf.epochs):
            train_result = train(train_loader, model, opt, scheduler, conf)
            val_result = validate(val_loader, model, conf)
            result = {
                'epoch': epoch,
                **train_result,
                **val_result,
            }
            hist.append(result)
            df_hist = pd.DataFrame(hist)

            log_hist(df_hist, conf.logger_epoch)
            write_on_board(df_hist, writer, conf)

            if epoch % 10 == 9:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            if conf.is_save_epoch_fn is not None and conf.is_save_epoch_fn(epoch):
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'writer_logdir': writer.logdir,
                }, f'{conf.out_dir}/{cv}-{epoch + 1:03d}.ckpt')
                df_hist.to_csv(f'{conf.out_dir}/{cv}.csv')
                print(f'Saved checkpoint {conf.out_dir}/{cv}-{epoch + 1:03d}.ckpt')

            should_stop = early_stopping.step(result['val_mean_lmae'])
            if should_stop:
                print(f'Early stopping at {epoch}')
                break

        df_hist = pd.DataFrame(hist)
        best = df_hist.sort_values('val_mean_lmae', ascending=True).head(1).iloc[0]
        print(best)

        writer.close()
        if conf.is_one_cv:
            break


# %%
main(Conf(
    is_one_cv=True,

    device='cuda',

    train_batch=32,
    val_batch=256,

    lr=1e-4,
    clr_max_lr=3e-3,
    clr_base_lr=3e-6,
    clr_gamma=0.999991,
    weight_decay=1e-4,

    tformer_dim=650,
    tformer_n_layers=14,
    tformer_d_inner=3800,
    tformer_dropout=0.03,
    tformer_dropatt=0.03,
    tformer_n_head=10,
    tformer_wnorm=True,

    epochs=400,

    db_path=const.DATA_DIR / 'artifacts' / 'data.pkl',

    exp_time=time(),
))
