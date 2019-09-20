import dataclasses
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean, scatter_add

from my_lib.torch.funcs import batched_index_select
from my_lib.torch.modules import MLP
from proj import const
from proj.loader import PandasDataset, atoms_collate_fn
from proj.util import get_scc_type_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
dataset = PandasDataset(pd.read_pickle(const.DATA_DIR / 'artifacts' / 'data.pkl'))
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=atoms_collate_fn,
)

# %%
MAX_BOND_COUNT = 250
NUM_BOND_ORIG_TYPES = 8


# %%
def loss(y_pred, y, x_bond):
    y_pred_pad = torch.cat([torch.zeros(y_pred.shape[0], 1, y_pred.shape[2], device=y_pred.device), y_pred], dim=1)

    # Note: The [:,:,1] below should match the num_bond_types[1]*final_dim in graph transformer
    y_pred_scaled = y_pred_pad.gather(1, x_bond[:, :, 1][:, None, :])[:, 0, :] * y[:, :, 2] + y[:, :, 1]
    abs_dy = (y_pred_scaled - y[:, :, 0]).abs()
    loss_bonds = (x_bond[:, :, 0] > 0)
    abs_err = abs_dy.masked_select(loss_bonds & (y[:, :, 3] > 0)).sum()

    type_dy = [abs_dy.masked_select(x_bond[:, :, 0] == i) for i in range(1, NUM_BOND_ORIG_TYPES + 1)]
    # if args.champs_loss:
    if False:
        type_err = torch.cat([t.sum().view(1) for t in type_dy], dim=0)
        type_cnt = torch.cat([torch.sum(x_bond[:, :, 0] == i).view(1) for i in range(1, NUM_BOND_ORIG_TYPES + 1)])
    else:
        type_err = torch.tensor([t.sum() for t in type_dy])
        type_cnt = torch.tensor([len(t) for t in type_dy])
    return abs_err, type_err, type_cnt


def calc_grouped_mae(y_pred, y_true, y_types, y_scaler):
    y_pred_scaled = y_pred.squeeze(dim=2) * y_scaler[:, :, 1] + y_scaler[:, :, 0]
    abs_err = (y_pred_scaled - y_true.squeeze(dim=2)).abs()
    mae_types = scatter_mean(abs_err.view(-1), y_types.view(-1))[1:]  # 0 is pad
    cnt_types = scatter_add(torch.ones_like(abs_err.view(-1)), y_types.view(-1))[1:]

    return mae_types, cnt_types


def sqdist(A, B):
    return (A ** 2).sum(dim=2)[:, :, None] + (B ** 2).sum(dim=2)[:, None, :] - 2 * torch.bmm(A, B.transpose(1, 2))


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


model = GraphTransformer(
    dim=650,
    # n_layers=14,
    n_layers=1,
    d_inner=3800,
    dropout=0.03,
    dropatt=0.0,
    n_head=10,
    wnorm=True,
).to(device)
model.eval()

# %%
for step, batch in enumerate(loader):
    batch = {
        k: v.to(device)
        for k, v in batch.items()
    }
    batch = AtomsData(**batch)

    with torch.no_grad():
        y_pred = model(batch)
        print(f'y_pred: {y_pred.shape}')

        mae_types, cnt_types = calc_grouped_mae(y_pred, batch.scc_val, batch.scc_type, batch.scc_scaler)

        enc = get_scc_type_encoder()
        for n, (mae, cnt) in enumerate(zip(mae_types, cnt_types)):
            print(const.TYPES[n], mae.item(), cnt.item())
        # print(b_abs_err, b_type_err, b_type_cnt)
    break
