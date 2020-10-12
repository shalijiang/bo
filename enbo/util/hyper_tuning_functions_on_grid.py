from numpy import genfromtxt
import torch.tensor as Tensor
import torch
from .benchmark_functions import BenchmarkFunction
from .utils import DataTransformer, standardize
import numpy as np
import pickle


class HyperTuningGrid(BenchmarkFunction):
    def __init__(
        self,
        dataname,
        datadir="./data",
        noise_std=None,
        maximize=False,
        normalize_x=True,
        normalize_y=True,
        interpolater=None,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dataname = dataname
        self.datapath = datadir + "/%s.csv" % dataname
        self.is_grid = True
        self.standardize = standardize

        data = genfromtxt(self.datapath, delimiter=",")
        is_not_nan = np.logical_not(np.isnan(data[:, -1]))
        self.x = Tensor(data[is_not_nan, :-1], device=device, dtype=dtype)
        self.y = Tensor(data[is_not_nan, -1], device=device, dtype=dtype)
        self.n, self.dim = self.x.shape
        self.device = device
        self.dtype = dtype
        self.original_bounds = Tensor(
            [[0.0, 1.0] for _ in range(self.dim)], device=device
        ).t()
        self.original_bounds[0] = torch.min(self.x, dim=0)[0]
        self.original_bounds[1] = torch.max(self.x, dim=0)[0]
        if normalize_x:
            transformer = DataTransformer(self.original_bounds)
            self.x = transformer.from_original_to_normalized(self.x)
            self.bounds = Tensor([[0.0, 1.0] for _ in range(self.dim)]).t()
        else:
            self.bounds = self.original_bounds

        if normalize_y:
            self.y = standardize(self.y)

        if interpolater is not None:
            model_path = datadir + "/" + dataname + ".model"
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
            except FileNotFoundError:
                self.model = interpolater.fit(self.x, self.y)
                with open(model_path, "wb") as f:
                    pickle.dump(self.model, f)
            self.interpolater = self.model.predict
            # self.interpolater = interpolater(self.x, self.y)
        else:
            self.interpolater = None
        self._optimal_value = self.y.min().item()
        min_idx = torch.argmin(self.y)
        self.x_star = self.x[min_idx]
        self.chosen_idx = None
        self._closest_point_index = -1

    def __str__(self):
        return (
            self.dataname
            + " n="
            + str(self.n)
            + " d="
            + str(self.dim)
            + " opt="
            + str(self.optimal_value)
        )

    def get_data(self, idx):
        return self.x[idx], self.y[idx] if not self.maximize else -self.y[idx]

    def set_chosen_idx(self, chosen_idx):
        self.chosen_idx = chosen_idx

    @property
    def unchosen_idx(self):
        unchosen_idx = torch.ones(self.n, dtype=torch.uint8, device=self.device)
        if self.chosen_idx is not None:
            unchosen_idx[self.chosen_idx] = 0
        unchosen_idx = torch.nonzero(unchosen_idx).squeeze()
        return unchosen_idx

    def find_closest(self, x):
        # find the point in the grid closest to x in Euclidean norm
        unchosen_idx = self.unchosen_idx
        distance2 = torch.sum((x.view(1, -1) - self.x[unchosen_idx]) ** 2, dim=1)
        index_to_unchosen = torch.argmin(distance2)
        index = unchosen_idx[index_to_unchosen]
        return index

    def evaluate_true(self, x):
        if self.interpolater is not None:
            return torch.tensor(self.interpolater(x), dtype=self.dtype)
        index = self.find_closest(x)
        self._closest_point_index = index
        return self.y[index]

    @property
    def closest_point_index(self):
        return self._closest_point_index
