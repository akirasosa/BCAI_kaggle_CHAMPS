import pandas as pd
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from proj import const
from proj.loader import PandasDataset, atoms_collate_fn
from proj.util import get_scc_type_encoder, get_all_df, get_structures_df

# %%
structs = pd.read_csv(Path(const.DATA_DIR, 'input', 'structures.csv'))

# %%
bonds = pd.read_csv(Path(const.DATA_DIR, 'processed', 'new_big_train.csv.bz2'))

# %%
atoms = pd.read_csv(Path(const.DATA_DIR, 'processed', 'new_big_structures.csv.bz2'))

# %%
atoms.atom.unique()

# %%
gamma = nn.Parameter(torch.ones(1)) * 2
gamma

# %%
tmp = torch.tensor([
    [
        [0, 0, 0],
        [1, 1, 1],
    ],
    [
        [2, 2, 2],
        [3, 3, 3],
    ],
]) + 10
idx = torch.tensor([
    [0, 0, 1, 1],
    [1, 1, 0, 0],
])

print(tmp.shape, idx.shape)


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out


batched_index_select(tmp, 1, idx)

# idx1, idx2 = idx.chunk(2, dim=2)

# %%
df_all = get_all_df()

# %%
scc_agg = df_all.groupby('type_encoded').agg({'scalar_coupling_constant': [np.mean, np.std]}).reset_index()
scc_agg.columns = ['type_encoded', 'scc_mean', 'scc_std']

# %%
df_all = df_all.merge(scc_agg, how='left', on='type_encoded')

# %%
data = df_all.head(1000)
mol_names = data.molecule_name.unique()
mol_pairs = data.groupby('molecule_name')

structures = get_structures_df()
mol_structures = structures.groupby('molecule_name')

# results = []


def mol_name_to_data(name):
    atoms = mol_structures.get_group(name)[[
        'atom_encoded',
        'x',
        'y',
        'z',
    ]].values

    scc = mol_pairs.get_group(name)[[
        'atom_index_0',
        'atom_index_1',
        'type_encoded',
        'scalar_coupling_constant',
        'scc_mean',
        'scc_std',
    ]].values

    return {
        'name': name,
        'atoms': atoms,
        'scc': scc,
    }


results = Parallel(n_jobs=1, verbose=1)([
    delayed(mol_name_to_data)(name)
    for name in mol_names
])

pd.DataFrame(results).to_pickle('/tmp/tmp.pkl')




