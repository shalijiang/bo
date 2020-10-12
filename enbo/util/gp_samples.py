from numpy import genfromtxt
import torch.tensor as Tensor
import torch
from .utils import DataTransformer, standardize
import numpy as np
from .hyper_tuning_functions_on_grid import HyperTuningGrid
from .benchmark_functions import BenchmarkFunction
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
import gpytorch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from torch.nn.functional import softplus
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from botorch.models.utils import validate_input_scaling


class GPSqeModel(BatchedMultiOutputGPyTorchModel, ExactGP):
    def __init__(
        self,
        train_X=None,
        train_Y=None,
        lengthscale=0.1,
        outputscale=4.0,
        noise_var=0.0001,
    ):
        if train_X is not None and train_Y is not None:
            validate_input_scaling(train_X=train_X, train_Y=train_Y)
            self._validate_tensor_args(X=train_X, Y=train_Y)
            self._set_dimensions(train_X=train_X, train_Y=train_Y)
            train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        ExactGP.__init__(self, train_X, train_Y, GaussianLikelihood())
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.covar_module.outputscale = Tensor(outputscale)
        self.covar_module.base_kernel.lengthscale = Tensor(lengthscale)
        self.likelihood = GaussianLikelihood()
        self.likelihood.noise = Tensor(noise_var)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPSamples(HyperTuningGrid, BenchmarkFunction):
    def __init__(self, seed, maximize=False, device=torch.device("cpu")):

        model = GPSqeModel()
        noise_std = model.likelihood.noise.sqrt().item()
        BenchmarkFunction.__init__(self, noise_std=None, maximize=maximize)
        self.model = model
        m = 100
        n = 100
        x1 = np.linspace(0.0, 1.0, m)
        x2 = np.linspace(0.0, 1.0, n)
        x1, x2 = np.meshgrid(x1, x2)
        X = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)
        self.x = Tensor(X, dtype=torch.float)
        model.eval()
        prior_dist = model(self.x)  # MultivariateNormal distribution of latent f
        torch.manual_seed(seed)
        with torch.no_grad():
            self.f = prior_dist.sample()
            self.y = self.f + torch.randn_like(self.f) * noise_std

        self.dataname = f"gp_sample{seed}"
        self.funcname = self.dataname
        self.is_grid = True

        self.n, self.dim = self.x.shape
        self.device = device
        self.original_bounds = Tensor(
            [[0.0, 1.0] for _ in range(self.dim)], device=device
        ).t()

        self.bounds = self.original_bounds

        self._optimal_value = self.y.min().item()
        min_idx = torch.argmin(self.y)
        self.x_star = self.x[min_idx]
        self.chosen_idx = None
        self._closest_point_index = -1
