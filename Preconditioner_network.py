import torch
from qcd_ml.nn.dense import v_Dense
from qcd_ml.nn.pt import v_PT


# define the preconditioner network
class Preconditioner_network(torch.nn.Module):
    def __init__(self, U, nr_layers, long_range_paths=False, smaller=False):
        super(Preconditioner_network, self).__init__()
        self.nr_layers = nr_layers

        paths = [[]]
        for mu in range(4):
            pathlength = 1
            max_pathlength = U.shape[mu + 1] // 2
            if smaller:
                max_pathlength //= 2
            while pathlength <= max_pathlength:
                paths.extend([[(mu, pathlength)], [(mu, -pathlength)]])
                pathlength *= 2
                if long_range_paths == False:
                    break
        nr_paths = len(paths)

        self.pt = v_PT(paths, U)
        self.dense_layers = torch.nn.ModuleList(
            [
                v_Dense(1, nr_paths),
                *[
                    v_Dense(nr_paths, nr_paths)
                    for _ in range(self.nr_layers - 1)
                ],
                v_Dense(nr_paths, 1),
            ]
        )

    def forward(self, v):
        vprev = torch.stack([v])
        for i in range(self.nr_layers + 1):
            v = self.dense_layers[i](vprev)
            v[0] += vprev[0]
            if i == self.nr_layers:
                break
            v = self.pt(v)
            vprev = v
        return v[0]
