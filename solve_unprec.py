import os

import numpy as np
import qcd_ml
import torch

from parameters import *

torch.manual_seed(42)

residuals_folder = residuals_folder_basename + "_unprec"

# define an inner product
innerproduct = lambda x, y: (x.conj() * y).sum()

# load the gauge field and define the Wilson operator
try:
    U = torch.load(config_file, weights_only=True)
except:
    raise RuntimeError("Loading of the gauge field failed")
w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, 1.0)

# lattice sizes
lattice_sizes = U.shape[1:5]

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
        eps=1e-8,
        maxiter=100000,
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
