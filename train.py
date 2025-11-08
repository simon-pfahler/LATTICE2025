import numpy as np
import qcd_ml
import torch

from parameters import *
from Preconditioner_network import Preconditioner_network

torch.manual_seed(42)

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

# create the model and initialize the weights
model = Preconditioner_network(U, nr_layers, long_range_paths)
for li in model.dense_layers:
    li.weights.data = 0.001 * torch.randn_like(
        li.weights.data, dtype=torch.cdouble
    )

# define the filtering function
pre_filter = lambda x: qcd_ml.util.solver.GMRES(
    w,
    torch.zeros_like(x),
    x,
    maxiter=filter_iteration,
    inner_iter=filter_iteration,
    eps=1e-15,
    innerproduct=innerproduct,
)[0]

# training
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
costs = np.zeros(training_steps)
print("Iteration - cost")
for t in range(training_steps):
    v = torch.randn(*lattice_sizes, 4, 3, dtype=torch.cdouble)
    if filter_iteration != 0:
        v = pre_filter(v)
    v /= torch.sqrt(innerproduct(v, v).real)

    diff = model.forward(w(v)) - v

    cost = innerproduct(diff, diff).real

    costs[t] = cost.item()
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print(f"{t} - {costs[t]}")
    # save weights
    torch.save(model.state_dict(), weights_filename)
    np.savetxt(history_filename, costs[: t + 1])

# save weights and history
torch.save(model.state_dict(), weights_filename)
np.savetxt(history_filename, costs)
