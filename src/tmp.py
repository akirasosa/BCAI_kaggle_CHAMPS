import pandas as pd
import gzip
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset, DataLoader

from graph_transformer import GraphTransformer
from modules.embeddings import LearnableEmbedding, SineEmbedding
from utils import const

mode = '_full'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
torch.ones((3,3)) * gamma
