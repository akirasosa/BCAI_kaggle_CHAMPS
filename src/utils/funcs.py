def batched_index_select(t, dim, indices):
    dummy = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out
