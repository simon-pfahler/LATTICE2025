# A novel gauge-equivariant neural network architecture for preconditioners in lattice QCD

This repository contains everything needed to recreate the results shown in the LATTICE 2025 talk "A novel gauge-equivariant neural network architecture for preconditioners in lattice QCD" by D. Knüttel, C. Lehner, **S. Pfahler**, T. Wettig.

## Prerequesties
The most important package that this code builds upon is [qcd_ml](https://github.com/daknuett/qcd_ml) [1]. Apart from that, only standard libraries are required (`torch`, `numpy`, `scipy` and `docopt`).

## How to use
The different data from the talk can be generated in the following ways:
- Spectra: run `bash makedirs.sh`, then run `python3.11 spectrum.py -l LATTICE -q Q -m M` with the desired values for lattice volume `LATTICE` ("8c16" or "16c32"), topological charge `Q` (possible values depend on `LATTICE`, see the available configurations in `configs`) and bare mass parameter `M`.
- Training networks: run `python3.11 train.py -l LATTICE -q Q -m M -f N -n L`, where `LATTICE`, `Q` and `M` are again the lattice volume, the topological charge and the bare mass parameter. `N` is the number of filter iterations in the filtered cost function, and `L` is the number of PT layers in the network.
- Test solves:
    - Unpreconditioned: `python3.11 solve_unprec.py -l LATTICE -q Q -m M`
    - Multigrid: `python3.11 solve_mg.py -l LATTICE -q Q -m M` (Note that the eigenvectors have to be available, so the corresponding spectrum needs to be calculated beforehand.)
    - Preconditioner network: `python3.11 solve_prec.py -l LATTICE -q Q -m M -f N -n L`
    - Preconditioner network (transfer): `python3.11 solve_prec_transfer.py -l LATTICE -q Q -m M -f N -n L` (the options describe the solved linear system. To specify which network to use, the correct path has to be provided in line 21 of `solve_prec_transfer.py`)

## Data availability
The 8c16 ensemble is available on Zenodo [2]. We use configurations 1200 (Q=0), 2200 (Q=1) and 2350 (Q=2).

The 16c32 ensemble will be avaiable on Zenodo shortly. We use configurations 282 (Q=0) and 329 (Q=4).

For ease of use, the configurations used in these tests are available in the folder `configs`.

## References
[1] D. Knuettel, S. Pfahler, Rielco, and R. Lehner, _daknuett/qcd_ml: v0.5.0_. (May 21, 2025). Zenodo. doi: [10.5281/zenodo.13254662](https://doi.org/10.5281/zenodo.13254662).

[2] D. Knüttel und D. Knüttel, _ldg/rqcd/qcdML/ens_001_. (March 13, 2025). Zenodo. doi: [10.5281/zenodo.15018324](https://doi.org/10.5281/zenodo.15018324).
