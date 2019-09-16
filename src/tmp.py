import gzip
import pickle
import torch

mode = '_full'
root_dir = '../processed'

with gzip.open(root_dir + f"/torch_proc_train{mode}_p1.pkl.gz", "rb") as f:
    print("Wait Patiently! Combining part 1 & 2 of the dataset so that we don't need to do it in the future.")
    D_train_part1 = pickle.load(f)
with gzip.open(root_dir + f"/torch_proc_train{mode}_p2.pkl.gz", "rb") as f:
    D_train_part2 = pickle.load(f)
D_train = tuple([torch.cat([D_train_part1[i], D_train_part2[i]], dim=0) for i in range(len(D_train_part1))])
with gzip.open(root_dir + '/' + f"/torch_proc_train{mode}.pkl.gz", "wb") as f:
    pickle.dump(D_train, f, protocol=4)
