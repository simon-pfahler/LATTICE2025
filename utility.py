from itertools import product

import torch

innerproduct = lambda x, y: (x.conj() * y).sum()


def orthonormalize(vecs):
    basis = torch.zeros_like(vecs)
    for i in range(vecs.shape[-1]):
        b_new = vecs[..., i]
        for j in range(i):
            b_new -= innerproduct(basis[..., j], b_new) * basis[..., j]
        basis[..., i] = b_new / torch.sqrt(innerproduct(b_new, b_new).real)
    return basis


def test_vectors(low_modes, block_sizes):
    lattice_sizes = low_modes.shape[:4]

    res = torch.zeros_like(low_modes)
    for bx, by, bz, bt in product(
        *[range(ll // bl) for ll, bl in zip(lattice_sizes, block_sizes)]
    ):
        slices = (
            slice(bx * block_sizes[0], (bx + 1) * block_sizes[0]),
            slice(by * block_sizes[1], (by + 1) * block_sizes[1]),
            slice(bz * block_sizes[2], (bz + 1) * block_sizes[2]),
            slice(bt * block_sizes[3], (bt + 1) * block_sizes[3]),
        )
        res[*slices] = orthonormalize(low_modes[*slices])
    return res
