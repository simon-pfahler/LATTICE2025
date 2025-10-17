from itertools import product

import torch

import utility

innerproduct = lambda x, y: (x.conj() * y).sum()


class MG:
    def __init__(self, low_modes, coarse_lattice_sizes):
        self.lattice_sizes = low_modes.shape[:4]
        self.coarse_lattice_sizes = coarse_lattice_sizes
        self.block_sizes = [
            l // cl
            for l, cl in zip(self.lattice_sizes, self.coarse_lattice_sizes)
        ]

        self.block_basis = torch.zeros_like(low_modes)
        for bx, by, bz, bt in product(
            *[range(bl) for bl in self.coarse_lattice_sizes]
        ):
            slices = (
                slice(bx * self.block_sizes[0], (bx + 1) * self.block_sizes[0]),
                slice(by * self.block_sizes[1], (by + 1) * self.block_sizes[1]),
                slice(bz * self.block_sizes[2], (bz + 1) * self.block_sizes[2]),
                slice(bt * self.block_sizes[3], (bt + 1) * self.block_sizes[3]),
            )
            self.block_basis[*slices] = utility.orthonormalize(
                low_modes[*slices]
            )

    def restrict(self, fine_vec):
        if fine_vec.shape != self.block_basis.shape[:-1]:
            raise ValueError(
                f"shape mismatch: got {fine_vec.shape} but "
                f"expected {self.block_basis.shape[:-1]}"
            )

        coarse_vec = torch.zeros(
            *self.coarse_lattice_sizes,
            self.block_basis.shape[-1],
            dtype=torch.complex128,
        )

        for bx, by, bz, bt in product(
            *[range(bl) for bl in self.coarse_lattice_sizes]
        ):
            slices = (
                slice(bx * self.block_sizes[0], (bx + 1) * self.block_sizes[0]),
                slice(by * self.block_sizes[1], (by + 1) * self.block_sizes[1]),
                slice(bz * self.block_sizes[2], (bz + 1) * self.block_sizes[2]),
                slice(bt * self.block_sizes[3], (bt + 1) * self.block_sizes[3]),
            )
            for k in range(self.block_basis.shape[-1]):
                coarse_vec[bx, by, bz, bt, k] = innerproduct(
                    self.block_basis[*slices, ..., k], fine_vec[*slices]
                )
        return coarse_vec

    def prolong(self, coarse_vec):
        if coarse_vec.shape != (
            *self.coarse_lattice_sizes,
            self.block_basis.shape[-1],
        ):
            raise ValueError(
                f"shape mismatch: got {coarse_vec.shape} but expected "
                "{(*self.coarse_lattice_sizes,self.block_basis.shape[-1])}"
            )

        fine_vec = torch.zeros_like(self.block_basis[..., 0])

        for bx, by, bz, bt in product(
            *[range(bl) for bl in self.coarse_lattice_sizes]
        ):
            slices = (
                slice(bx * self.block_sizes[0], (bx + 1) * self.block_sizes[0]),
                slice(by * self.block_sizes[1], (by + 1) * self.block_sizes[1]),
                slice(bz * self.block_sizes[2], (bz + 1) * self.block_sizes[2]),
                slice(bt * self.block_sizes[3], (bt + 1) * self.block_sizes[3]),
            )
            fine_vec[*slices] = sum(
                coarse_vec[bx, by, bz, bt, k]
                * self.block_basis[*slices, ..., k]
                for k in range(self.block_basis.shape[-1])
            )
        return fine_vec
