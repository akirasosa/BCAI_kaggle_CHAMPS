import gzip
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset, DataLoader

from graph_transformer import GraphTransformer
from utils import const

mode = '_full'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
with gzip.open(Path(const.DATA_DIR, 'processed', f"torch_proc_train{mode}_p1.pkl.gz"), "rb") as f:
    print("Wait Patiently! Combining part 1 & 2 of the dataset so that we don't need to do it in the future.")
    D_train_part1 = pickle.load(f)

dataset = TensorDataset(*D_train_part1)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# %%
MAX_BOND_COUNT = 250
NUM_BOND_ORIG_TYPES = 8


# %%
# noinspection PyUnreachableCode
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


def sqdist(A, B):
    return (A ** 2).sum(dim=2)[:, :, None] + (B ** 2).sum(dim=2)[:, None, :] - 2 * torch.bmm(A, B.transpose(1, 2))


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


# noinspection PyShadowingNames
class GraphTransformer(nn.Module):
    def __init__(self, dim, n_layers, d_inner,
                 dropout=0.0,
                 dropatt=0.0,
                 n_head=10,
                 wnorm=False):
        super().__init__()

        self.atom_embedding = nn.Embedding(5, dim)
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

    def forward(self, x_atom, x_atom_pos):
        # PART I: Form the embeddings and the distance matrix
        D = sqdist(x_atom_pos[:, :, :3],
                   x_atom_pos[:, :, :3]).to(device)

        mask = x_atom[:, :, 0] > 0
        mask = torch.einsum('bi, bj->bij', mask, mask).type(x_atom_pos.dtype)

        new_mask = -1e20 * torch.ones_like(mask).to(mask.device)
        new_mask[mask > 0] = 0
        print(1, mask.dtype)

        Z = self.atom_embedding(x_atom[:, :, 0])

        # PART II: Pass through a bunch of self-attention and position-wise feed-forward blocks
        for i in range(len(self.layers)):
            Z = self.layers[i](Z, D, new_mask, mask, store=False)

        self.apply(self.weights_init)

        return Z

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
    batch = [
        item.to(device)
        for n, item in enumerate(batch)
    ]
    x_atom = batch[1]
    x_atom_pos = batch[2]
    x_bond = batch[3]
    x_bond_dist = batch[4]
    x_triplet = batch[5]
    x_triplet_angle = batch[6]
    x_quad = batch[7]
    x_quad_angle = batch[8]
    y = batch[9]

    x_bond, x_bond_dist, y = x_bond[:, :MAX_BOND_COUNT], x_bond_dist[:, :MAX_BOND_COUNT], y[:, :MAX_BOND_COUNT]

    print(f'x_atom: {x_atom.shape}')
    print(x_atom)
    # print(f'y: {y.shape}')

    with torch.no_grad():
        y_pred, _ = model(x_atom, x_atom_pos)
        print(f'y_pred: {y_pred.shape}')
        # b_abs_err, b_type_err, b_type_cnt = loss(y_pred, y, x_bond)
        # print(b_abs_err, b_type_err, b_type_cnt)

    if step == 0:
        break

# %%
a = torch.tensor([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
]).unsqueeze(dim=0)
print(a.shape)
sqdist(a, a)

# %%
mask = torch.tensor([
    [0, 1, 1],
    [1, 0, 1],
])
mask = torch.einsum('bi, bj->bij', mask, mask)
mask

# %%
new_mask = -1e20 * torch.ones_like(mask).to(mask.device)
new_mask[mask > 0] = 0
new_mask
