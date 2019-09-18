from pathlib import Path

import pandas as pd

from utils import const

# %%
structs = pd.read_csv(Path(const.DATA_DIR, 'input', 'structures.csv'))

# %%
structs.groupby('molecule_name').size().max()
