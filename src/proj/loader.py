import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from my_lib.torch.dataset import PandasDataset
from proj import const


def pad_2d(a, shape):
    result = np.zeros(shape)
    result[:a.shape[0], :a.shape[1]] = a

    return result


def atoms_collate_fn(examples):
    atom_shapes = np.array([e['atoms'].shape for e in examples])
    n_max_atoms = atom_shapes[:, 0].max()
    n_atom_cols = atom_shapes[:, 1].max()
    atoms = np.array([
        pad_2d(e['atoms'], (n_max_atoms, n_atom_cols))
        for e in examples
    ])

    pair_shapes = np.array([e['scc'].shape for e in examples])
    n_max_pairs = pair_shapes[:, 0].max()
    n_pair_cols = pair_shapes[:, 1].max()
    scc = np.array([
        pad_2d(e['scc'], (n_max_pairs + 1, n_pair_cols))  # make it always have at least one pad.
        for e in examples
    ])

    return {
        'atom_type': torch.from_numpy(atoms[:, :, :1].astype(int)),
        'atom_pos': torch.from_numpy(atoms[:, :, 1:4].astype(np.float32)),
        'scc_idx': torch.from_numpy(scc[:, :, :2].astype(int)),
        'scc_type': torch.from_numpy(scc[:, :, 2:3].astype(int)),
        'scc_val': torch.from_numpy(scc[:, :, 3:4].astype(np.float32)),
        'scc_scaler': torch.from_numpy(scc[:, :, 4:6].astype(np.float32)),
    }


if __name__ == '__main__':
    df = pd.read_pickle(const.DATA_DIR / 'artifacts' / 'data.pkl')
    dataset = PandasDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=atoms_collate_fn,
    )
    for batch in loader:
        # pass
        print(batch['atom_type'].shape)
        print(batch['scc_type'])
        break
