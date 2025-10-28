import os
from itertools import product

import numpy as np
import qcd_ml
import torch

from MG_preconditioner import MG
from parameters import *

torch.manual_seed(42)

residuals_folder = residuals_folder_basename + "_mg"

# define an inner product
innerproduct = lambda x, y: (x.conj() * y).sum()

# load the gauge field and define the Wilson operator
try:
    U = torch.load(config_file, weights_only=True)
except:
    raise RuntimeError("Loading of the gauge field failed")
w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)

# load the eigenvectors
try:
    eigenvectors = np.load(ev_file)
    eigenvectors = torch.from_numpy(eigenvectors[..., :12])
except:
    raise RuntimeError("Loading of the eigenvectors failed")

# lattice sizes
lattice_sizes = U.shape[1:5]
coarse_lattice_sizes = [l // 4 for l in lattice_sizes]

# >>> generate new vectors
# for i in range(12):
#    xinit = torch.randn(*lattice_sizes, 4, 3, dtype=torch.cdouble)
#    eigenvectors[..., i], _ = qcd_ml.util.solver.GMRES(
#        w, torch.zeros_like(xinit), xinit, eps=1e-5
#    )
# <<< generate new vectors

# define MG
mg = MG(eigenvectors, coarse_lattice_sizes)
w_coarse = lambda x: mg.restrict(w(mg.prolong(x)))

# get precomputed w coarse
w_coarse_precomputed = torch.zeros(
    (*coarse_lattice_sizes, 12, *coarse_lattice_sizes, 12), dtype=torch.cdouble
)

try:
    w_coarse_precomputed = torch.load(
        f"./wcoarse/{lattice}/Q{Q}.pt", weights_only=True
    )
except:
    for bx, by, bz, bt in product(*[range(bl) for bl in coarse_lattice_sizes]):
        print(f"Calculating precomupted w_coarse for {bx},{by},{bz},{bt}")
        for idx in range(12):
            rhs = torch.zeros((*coarse_lattice_sizes, 12), dtype=torch.cdouble)
            rhs[bx, by, bz, bt, idx] = 1
            w_coarse_precomputed[..., bx, by, bz, bt, idx] = w_coarse(rhs)
    torch.save(w_coarse_precomputed, f"./wcoarse/{lattice}/Q{Q}.pt")

w_coarse_precomputed_lambda = lambda x: torch.einsum(
    "xyztiabcdj,abcdj->xyzti", w_coarse_precomputed, x
)


def prec(vec):
    # pre-smoothing
    if mg_presmoothing_steps > 0:
        presmoothed = qcd_ml.util.solver.GMRES(
            w, vec, torch.clone(vec), maxiter=mg_smoothing_steps
        )[0]
    else:
        presmoothed = torch.clone(vec)

    # get coarse grid residual
    fine_residual = vec - w(presmoothed)
    coarse_residual = mg.restrict(fine_residual)

    # coarse-grid correction
    cgc_coarse_vec = qcd_ml.util.solver.GMRES(
        w_coarse_precomputed_lambda,
        coarse_residual,
        torch.clone(coarse_residual),
        eps=1e-5,
        maxiter=10000,
    )[0]

    # move coarse-grid correction back to fine grid
    fine_improved = presmoothed + mg.prolong(cgc_coarse_vec)

    # post-smoothing
    if mg_postsmoothing_steps > 0:
        postsmoothed = qcd_ml.util.solver.GMRES(
            w, vec, fine_improved, maxiter=mg_postsmoothing_steps
        )[0]
    else:
        postsmoothed = torch.clone(fine_improved)

    return postsmoothed


# generate right-hand sides
torch.manual_seed(43)
rhs_vectors = [
    torch.randn(*lattice_sizes, 4, 3, dtype=torch.cdouble)
    for _ in range(nr_solves)
]
rhs_vectors = [e / innerproduct(e, e) ** 0.5 for e in rhs_vectors]

# create output directory
try:
    os.makedirs(residuals_folder)
except:
    raise FileExistsError("Residuals folder already exists")

# solve Dirac equation
its = np.zeros(nr_solves)
for i, rhs_vector in enumerate(rhs_vectors):
    _, ret_p = qcd_ml.util.solver.GMRES(
        w,
        rhs_vector,
        torch.zeros_like(rhs_vector),
        preconditioner=prec,
        eps=1e-8,
        maxiter=10000,
        verbose=True,
    )
    its[i] = ret_p["k"]
    np.savetxt(
        os.path.join(residuals_folder, f"residuals_sample{i}.dat"),
        ret_p["history"],
    )
print(
    f"Model Iteration count: {np.mean(its)} +- "
    + f"{np.std(its, ddof=1)/np.sqrt(len(rhs_vectors))}"
)
