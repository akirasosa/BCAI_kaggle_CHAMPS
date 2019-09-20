import numpy as np
import pandas as pd
from tqdm import tqdm

from proj import const
from proj.util import get_all_df, get_structures_df

# %%
df_all = get_all_df()

# %%
scc_agg = df_all.groupby('type_encoded').agg({'scalar_coupling_constant': [np.mean, np.std]}).reset_index()
scc_agg.columns = ['type_encoded', 'scc_mean', 'scc_std']

# %%
df_all = df_all.merge(scc_agg, how='left', on='type_encoded')

# %%
# data = df_all.head(1000)
data = df_all
mol_names = data.molecule_name.unique()
mol_pairs = data.groupby('molecule_name')

structures = get_structures_df()
mol_structures = structures.groupby('molecule_name')


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
        'is_test': np.isnan(scc).any(),
    }


# results = Parallel(n_jobs=1, verbose=2)([
#     delayed(mol_name_to_data)(name)
#     for name in mol_names
# ])
results = [
    mol_name_to_data(name)
    for name in tqdm(mol_names)
]

df = pd.DataFrame(results)
data_path = const.DATA_DIR / 'artifacts' / 'data.pkl'
df.to_pickle(data_path)
print(f'Saved data at {data_path} .')

# %%
df[['is_test']].head()
