"""
Usage:
    script.py [options]

Options:
    -m --mass=MASS          Bare mass [default: 0.0]
    -n --nr-layers=LAYERS   Number of layers [default: 4]
    -q --q=Q                Nr of topological modes [default: 0]
    -f --filter=ITER        Filter iteration [default: 10]
    -l --lattice=LATTICE    Lattice volume [default: 8c16]
"""

from docopt import docopt

# parse command line arguments
args = docopt(__doc__)

Q = int(args["--q"])
filter_iteration = int(args["--filter"])
lattice = args["--lattice"]
mass = float(args["--mass"])
nr_layers = int(args["--nr-layers"])

# Preconditioner network parameters
long_range_paths = True

# training parameters
learning_rate = 1e-2
training_steps = 1000

# solve parameters
mg_presmoothing_steps = 10
mg_postsmoothing_steps = 10
nr_solves = 10

# Wilson operator paths
config_file = f"./configs/{lattice}/Q{Q}.pt"
ev_file = f"./eigenvectors/{lattice}/Q{Q}.npy"

# paths
history_filename = f"history_{lattice}_Q{Q}_{nr_layers}layers_mass{mass}_filter{filter_iteration}.dat"
weights_filename = f"weights_{lattice}_Q{Q}_{nr_layers}layers_mass{mass}_filter{filter_iteration}.pt"
if long_range_paths:
    history_filename = f"history_{lattice}_Q{Q}_{nr_layers}layers_long_range_mass{mass}_filter{filter_iteration}.dat"
    weights_filename = f"weights_{lattice}_Q{Q}_{nr_layers}layers_long_range_mass{mass}_filter{filter_iteration}.pt"
residuals_folder_basename = f"residuals_{lattice}_Q{Q}_mass{mass}"
