import sys

import numpy as np
import qcd_ml
import torch
from scipy.sparse.linalg import LinearOperator, eigs

from parameters import *

# define an inner product
innerproduct = lambda x, y: (x.conj() * y).sum()

# load the gauge field and define the Wilson operator
try:
    U = torch.load(config_file, weights_only=True)
except:
    raise RuntimeError("Loading of the gauge field failed")
w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)

Gamma5 = torch.eye(4, 4, dtype=torch.cdouble)
for g in qcd_ml.qcd.static.gamma:
    Gamma5 = torch.einsum("ij,jk->ik", Gamma5, g)

# lattice sizes
lattice_sizes = U.shape[1:5]


# apply Dirac operator
def w_np(x):
    inp = torch.tensor(
        np.reshape(x, (*lattice_sizes, 4, 3)), dtype=torch.cdouble
    )
    res = w(inp)
    return np.reshape(res.detach().numpy(), (np.prod(lattice_sizes) * 4 * 3))


# apply inverse of Dirac operator
def winv_np(x):
    inp = torch.tensor(
        np.reshape(x, (*lattice_sizes, 4, 3)), dtype=torch.cdouble
    )
    res, _ = qcd_ml.util.solver.GMRES(
        w,
        inp,
        torch.clone(inp),
        eps=1e-8,
    )
    print(len(_["history"]))
    return np.reshape(res.detach().numpy(), (np.prod(lattice_sizes) * 4 * 3))


winv_LinOp = LinearOperator(
    shape=(np.prod(lattice_sizes) * 4 * 3, np.prod(lattice_sizes) * 4 * 3),
    matvec=lambda x: winv_np(x),
)
k = 12
eigenvalues, eigenvectors = eigs(
    winv_LinOp, k=k, which="LM", return_eigenvectors=True, tol=1e-8
)

# get eigenvalues of w instead of winv
eigenvalues = 1.0 / eigenvalues

# make sure sorting is correct
sorted_eigenvalues_indices = np.argsort(np.abs(eigenvalues))
eigenvalues = np.array(eigenvalues)[sorted_eigenvalues_indices]
eigenvectors = eigenvectors[:, sorted_eigenvalues_indices]

print(
    f"Lowest {k} eigenvalues of the preconditioned system (Re Im Tolerance Chirality):"
)
errs = list()
chiralities = list()
for i, eigenvalue in enumerate(eigenvalues):
    eigenvector = eigenvectors[:, i]
    applied_eigenvector = w_np(eigenvector)
    scaled_eigenvector = eigenvalue * eigenvector
    errs.append(np.linalg.norm(applied_eigenvector - scaled_eigenvector))
    eigenvector = np.reshape(eigenvector, (*lattice_sizes, 4, 3))
    chiralities.append(
        innerproduct(
            np.einsum("st,...tc->...sc", Gamma5.numpy(), eigenvector),
            eigenvector,
        ).real
    )

# move eigenvalues so they are given for mass 0
eigenvalues -= mass

for i, eigenvalue in enumerate(eigenvalues):
    print(eigenvalue.real, eigenvalue.imag, errs[i], chiralities[i])

# save eigenvectors
eigenvectors = np.reshape(eigenvectors, (*lattice_sizes, 4, 3, k))
np.save(f"{ev_file[:-4]}.npy", eigenvectors[..., :12])
