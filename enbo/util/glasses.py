import torch
from botorch.models.model import Model
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement
from torch import Tensor
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from typing import Optional


def glasses(
    x: Tensor,
    model: Model,
    bounds: Tensor,
    L: float,
    y0: float,
    horizon: int = 1,
    num_mc_samples: int = 10000,
    sample_seed: Optional[int] = None,
):
    """
    Compute the GLASSES acquisition function proposed in Gonzalez et al.,
    Batch Bayesian Optimization via Local Penalization, AISTATS 2016.

    Assume maximization problem.

    Input:
        x: Tensor, point to evaluate [currently only support evaluating a single point]
        model: GP model, containing current observations
        bounds: Tensor, optimization domain,
                e.g., [-1, 1]^2 for a 2D function
        L: float, an estimate of the Lipschitz constant of the target function
        y0: float, current best observed value
        horizon: integer, number of steps to lookahead,
                e.g., the remaining budget
        num_mc_samples: integer,
            number of quasi Monte Carlo samples to estimate qEI
        sample_seed: None or int, random seed for sampling y

    Output:
        a scalar, the GLASSES utility of x
    """
    X = predict_future_locations(x, model, horizon, bounds, L, y0)
    qmc_sampler = SobolQMCNormalSampler(num_samples=num_mc_samples, seed=sample_seed)
    qEI = qExpectedImprovement(model=model, best_f=y0, sampler=qmc_sampler)
    return qEI(X)


def predict_future_locations(
    x, model, horizon, bounds, L, M, num_restarts=2, raw_samples=100, options=None
):
    X = [x] + [0] * (horizon - 1)
    ei = ExpectedImprovement(model=model, best_f=M)

    for i in range(1, horizon):

        def penalizer(x):
            res = 1
            for j in range(i):
                res = res * local_penalizer(x, X[j], model, L, M)
            return res

        def penalized_ei(x):
            return ei(x) * penalizer(x)

        x_i, _ = optimize_acqf(
            acq_function=penalized_ei,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options or {},
        )
        X[i] = x_i.squeeze()
    return torch.stack(X)


def local_penalizer(x, x_j, model, L, M):
    threshold = M - L * (x - x_j).norm()
    posterior = model.posterior(x_j.unsqueeze(0))
    normal = torch.distributions.Normal(
        posterior.mean.item(), posterior.variance.sqrt().item()
    )
    return 1 - normal.cdf(threshold)


class MeanGradientL2(AcquisitionFunction):
    @torch.enable_grad()
    def forward(self, x):
        x.requires_grad = True
        posterior_mean = self.model.posterior(x).mean
        x_grad = torch.autograd.grad(
            posterior_mean.squeeze(),
            x,
            grad_outputs=torch.Tensor([1.0] * len(posterior_mean)),
            create_graph=True,
        )[0]
        return x_grad.norm(dim=-1).squeeze()


def estimate_lipschitz_constant(
    model, bounds, num_restarts=20, raw_samples=512, options=None
):
    # return 1.0  # mock test
    mean_gradient_l2 = MeanGradientL2(model)
    _, L = optimize_acqf(
        acq_function=mean_gradient_l2,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options or {},
    )

    return L
