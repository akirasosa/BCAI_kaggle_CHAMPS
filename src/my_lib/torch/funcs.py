import torch


def sqdist(A, B):
    return (A ** 2).sum(dim=2)[:, :, None] + (B ** 2).sum(dim=2)[:, None, :] - 2 * torch.bmm(A, B.transpose(1, 2))
