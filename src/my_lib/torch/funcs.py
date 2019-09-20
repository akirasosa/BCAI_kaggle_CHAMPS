import torch


def sqdist(A, B):
    return (A ** 2).sum(dim=2)[:, :, None] + (B ** 2).sum(dim=2)[:, None, :] - 2 * torch.bmm(A, B.transpose(1, 2))


def batched_index_select(t, dim, indices):
    dummy = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out
