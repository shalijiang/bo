import torch
from botorch.models.model import Model
from botorch.acquisition.analytic import ExpectedImprovement
from torch import Tensor
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from copy import deepcopy
from typing import Union, Tuple, Optional
import numpy as np
import math
import scipy


CC = "continuous-continuous"
CG = "continuous-grid"
GG = "grid-grid"
GC = "grid-continous"


def rollout(
    x: Tensor,
    model: Model,
    best_f: Union[float, Tensor],
    bounds: Tensor,
    mode: str = CC,
    x_grid: Optional[Tensor] = None,  # if given, optimize on grid
    idx: Union[int, Tensor] = None,  # if given x will be overwrite by x_grid[idx]
    quadrature: Union[str, Tuple] = "qmc",
    horizon: int = 4,
    num_y_samples: int = 5,
):
    """
    continuous domain rollout, expectation estimated using  (quasi) Monte Carlo or Gaussian-Hermite quadrature
    EI_rollout(x) = E_y[ max(y-y0,0) + EI_rollout(x'| x, y) ], where x'=argmax EI(x' | x, y)
    define f(y) = max(y-y0,0) + EI_rollout(x'|x,y)
    then
    EI_rollout(x) = \int w(y) f(y) dy
    where the weight function w(y) is a Gaussian density function N(mu, sigma^2)
    We can estimate this integral using quasi Monte Carlo samples from w(y)
    or use Gauss-Hermite quadrature, as in Lam et al. (2016):
    such a integration can be transformed into the standard Gaussian-Hermite quadrature formulation
    EI_rollout(x) = 1/sqrt(pi) \int exp(-t^2) f(sqrt(2)*sigma*t+mu) dt,  where t = (y-mu)/sqrt(2)/sigma

    We first generate Gauss-Hermite quadrature sample locations t_i and weights w_i using numpy.polynomial.hermite.hermgauss
    then estimate the expectation by
    EI_rollout(x) \approx 1/sqrt(pi) \sum_i w_i f(sqrt(2)*sigma*t_i +mu)

    :param x: a single point
    :param model: the GP model
    :param best_f: current best observed value
    :param bounds: bounds of the domain, shape (2, d)
    :param base_acquisition:
    :param quadrature: Monte Carlo or Quasi Monte Carlo
    :param horizon: rollout horizon
    :param num_y_samples:  number of (quasi) Monte Carlo samples for estimating the integral
    :return:
    """

    if mode == GG:
        x = x_grid[idx]
    with torch.no_grad():
        acq_func = ExpectedImprovement(model=model, best_f=best_f)
        one_step_improvement = acq_func(x.unsqueeze(0)).item()
        if horizon <= 1:
            return one_step_improvement

        # compute posterior
        posterior = model.posterior(x.unsqueeze(0))
        if isinstance(quadrature, str) and quadrature == "qmc":  # quasi Monte Carlo
            with torch.no_grad():
                sampler = SobolQMCNormalSampler(num_samples=num_y_samples)
                samples = sampler(posterior).squeeze().numpy()
            weights = torch.ones(num_y_samples) / num_y_samples

        elif isinstance(quadrature, Tuple):
            mu = posterior.mean.item()
            sigma = torch.sqrt(posterior.variance).item()
            samples, weights = np.polynomial.hermite.hermgauss(num_y_samples)
            samples = np.sqrt(2.0) * sigma * samples + mu
            weights /= np.sqrt(math.pi)

    future_reward_samples = np.zeros(num_y_samples)
    for i in range(num_y_samples):
        y_sample = samples[i]

        fake_model: Model = deepcopy(model)
        x0 = model.train_inputs[0].squeeze(0)
        y0 = model.train_targets.squeeze(0)
        train_x = torch.cat((x0, x.unsqueeze(0)), -2)
        train_y = torch.cat((y0, Tensor([y_sample])))
        fake_model.set_train_data(inputs=train_x, targets=train_y, strict=False)
        best_f_new = max(best_f, y_sample)  # maximization problem
        acq_func = ExpectedImprovement(model=fake_model, best_f=best_f_new)

        if mode == CC:
            # optimize on continuous domain, rollout on continuous domain
            options = {
                "simple_init": True,
                "maxiter": 100,
                "seed": i,
            }  # add seed to ensure sobol sequence is reproduciable
            next_x, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=2,
                raw_samples=100,
                options=options,
            )

            future_reward_samples[i] = rollout(
                next_x.squeeze(),
                model=fake_model,
                best_f=best_f_new,
                bounds=bounds,
                mode=mode,
                quadrature=quadrature,
                horizon=horizon - 1,
                num_y_samples=num_y_samples,
            )
        elif mode == GG:
            # the function is a grid function
            # optimize on the grid, also rollout on the grid
            ei_values = acq_func(x_grid.unsqueeze(1))
            next_idx = torch.argmax(ei_values)
            future_reward_samples[i] = rollout(
                None,
                model=fake_model,
                best_f=best_f_new,
                bounds=bounds,
                mode=mode,
                x_grid=x_grid,
                idx=next_idx,
                quadrature=quadrature,
                horizon=horizon - 1,
                num_y_samples=num_y_samples,
            )
        elif mode == CG:
            # the function is a continuous function, but only evaluated on a grid
            # in this case, only rollout on the grid points
            ei_values = acq_func(x_grid.unsqueeze(1))
            next_idx = torch.argmax(ei_values)
            next_x = x_grid[next_idx]
            future_reward_samples[i] = rollout(
                next_x,
                model=fake_model,
                best_f=best_f_new,
                bounds=bounds,
                mode=mode,
                x_grid=x_grid,
                quadrature=quadrature,
                horizon=horizon - 1,
                num_y_samples=num_y_samples,
            )
        elif mode == GC:
            raise NotImplementedError
        else:
            print("unknown mode")

    future_reward = future_reward_samples.dot(weights)

    return one_step_improvement + future_reward


def rollout_wrapper(
    x: Tensor,
    model: Model,
    best_f: Union[float, Tensor],
    bounds: Tensor,
    x_grid: Optional[Tensor] = None,  # if given, optimize on grid
    indices: Union[int, Tensor] = None,  # if given x will be overwrite by x_grid[idx]
    quadrature: Union[str, Tuple] = "qmc",
    horizon: int = 4,
    num_y_samples: int = 5,
):
    n = len(indices)
    values = np.zeros(n)
    for i in range(n):
        idx = indices[i]
        values[i] = rollout(
            None,
            model=model,
            best_f=best_f,
            bounds=bounds,
            mode=GG,
            x_grid=x_grid,
            idx=idx,
            quadrature=quadrature,
            horizon=horizon,
            num_y_samples=num_y_samples,
        )
    return values


class UserData(object):
    def __init__(
        self, model, best_f, bounds, mode, x_grid, quadrature, horizon, num_y_samples
    ):
        self.model = model
        self.best_f = best_f
        self.bounds = bounds
        self.mode = mode
        self.x_grid = x_grid
        self.quadrature = quadrature
        self.horizon = horizon
        self.num_y_samples = num_y_samples


def rollout_wrapper_direct(x, user_data: UserData):
    value = rollout(
        Tensor(x),
        model=user_data.model,
        best_f=user_data.best_f,
        bounds=user_data.bounds,
        mode=user_data.mode,
        x_grid=user_data.x_grid,
        quadrature=user_data.quadrature,
        horizon=user_data.horizon,
        num_y_samples=user_data.num_y_samples,
    )
    return -value, 0  # DIRECT.solve minimization problem


def rollout_quad(
    x: Tensor,
    model: Model,
    best_f: Union[float, Tensor],
    bounds: Tensor,
    horizon: int = 4,
):
    """
    continuous domain rollout, use adaptive quadrature
    """
    if horizon == 1:
        return ExpectedImprovement(model=model, best_f=best_f)(x).item()

    # compute posterior
    posterior = model.posterior(x)
    mu = posterior.mean.item()
    sigma = posterior.variance.sqrt().item()

    def integrand(y_sample):
        if not isinstance(y_sample, np.float):
            n = len(y_sample)
        else:
            n = 1
            y_sample = np.array([y_sample])
        res = np.zeros(n)
        for i in range(n):
            y = y_sample[i]
            one_step_improvement = max(y - best_f, 0)
            fake_model: Model = deepcopy(model)
            x0 = model.train_inputs[0]
            y0 = model.train_targets
            train_x = torch.cat([x0, x], -2)
            train_y = torch.cat([y0, Tensor([y])], -1)
            fake_model.reinitialize(train_X=train_x, train_Y=train_y)
            best_f_new = max(best_f, y)  # maximization problem

            acq_func = ExpectedImprovement(model=fake_model, best_f=best_f_new)
            options = {
                "simple_init": True,
                "maxiter": 500,
                "seed": 0,
            }  # add seed to ensure sobol sequence is reproduciable
            new_x = joint_optimize(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=3,
                raw_samples=10,
                options=options,
            )

            future_reward = rollout_quad(
                new_x,
                model=fake_model,
                best_f=best_f_new,
                bounds=bounds,
                horizon=horizon - 1,
            )
            total_improvement = one_step_improvement + future_reward
            res[i] = total_improvement * scipy.stats.norm.pdf(y, loc=mu, scale=sigma)
        return res

    lower_limit = mu - 4 * sigma
    upper_limit = mu + 4 * sigma
    value, error = scipy.integrate.quadrature(integrand, lower_limit, upper_limit)

    return value


def rollout_discrete(
    x_grid: Tensor,
    idx: Union[int, Tensor],
    model: Model,
    best_f: Union[float, Tensor],
    bounds: Tensor,
    quadrature: Union[str, Tuple] = "qmc",
    horizon: int = 4,
    num_y_samples: int = 10,
):
    """
    continuous domain rollout, expectation estimated using  (quasi) Monte Carlo or Gaussian-Hermite quadrature
    EI_rollout(x) = E_y[ max(y-y0,0) + EI_rollout(x'| x, y) ], where x'=argmax EI(x' | x, y)
    define f(y) = max(y-y0,0) + EI_rollout(x'|x,y)
    then
    EI_rollout(x) = \int w(y) f(y) dy
    where the weight function w(y) is a Gaussian density function N(mu, sigma^2)
    We can estimate this integral using quasi Monte Carlo samples from w(y)
    or use Gauss-Hermite quadrature, as in Lam et al. (2016):
    such a integration can be transformed into the standard Gaussian-Hermite quadrature formulation
    EI_rollout(x) = 1/sqrt(pi) \int exp(-t^2) f(sqrt(2)*sigma*t+mu) dt,  where t = (y-mu)/sqrt(2)/sigma

    We first generate Gauss-Hermite quadrature sample locations t_i and weights w_i using numpy.polynomial.hermite.hermgauss
    then estimate the expectation by
    EI_rollout(x) \approx 1/sqrt(pi) \sum_i w_i f(sqrt(2)*sigma*t_i +mu)

    :param x: a single point
    :param model: the GP model
    :param best_f: current best observed value
    :param bounds: bounds of the domain, shape (2, d)
    :param base_acquisition:
    :param quadrature: Monte Carlo or Quasi Monte Carlo
    :param horizon: rollout horizon
    :param num_y_samples:  number of (quasi) Monte Carlo samples for estimating the integral
    :return:
    """

    if horizon == 1:
        acq_func = ExpectedImprovement(model=model, best_f=best_f)
        return acq_func(x).item()

    x = x_grid[idx]
    # compute posterior
    posterior = model.posterior(x)
    if isinstance(quadrature, str) and quadrature == "qmc":  # quasi Monte Carlo
        with torch.no_grad():
            sampler = SobolQMCNormalSampler(num_samples=num_y_samples)
            samples = sampler(posterior).squeeze().numpy()
        weights = torch.ones(num_y_samples) / num_y_samples

    elif isinstance(quadrature, Tuple):
        mu = posterior.mean.item()
        sigma = torch.sqrt(posterior.variance).item()
        samples, weights = np.polynomial.hermite.hermgauss(num_y_samples)
        samples = np.sqrt(2.0) * sigma * samples + mu
        weights /= np.sqrt(math.pi)

    improvement_of_samples = np.zeros(num_y_samples)
    for i in range(num_y_samples):
        y_sample = samples[i]
        one_step_improvement = max(y_sample - best_f, 0)

        fake_model: Model = deepcopy(model)
        x0 = model.train_inputs[0]
        y0 = model.train_targets
        train_x = torch.cat([x0, x.unsqueeze(0)], -2)
        train_y = torch.cat([y0, Tensor([y_sample])])
        fake_model.reinitialize(train_X=train_x, train_Y=train_y)
        best_f_new = max(best_f, y_sample)  # maximization problem
        acq_func = ExpectedImprovement(model=fake_model, best_f=best_f_new)
        ei_values = acq_func(X)
        idx = torch.argmax(ei_values)

        future_reward = rollout(
            x_grid,
            idx,
            model=fake_model,
            best_f=best_f_new,
            bounds=bounds,
            quadrature=quadrature,
            horizon=horizon - 1,
            num_y_samples=num_y_samples,
        )
        improvement_of_samples[i] = one_step_improvement + future_reward

    return improvement_of_samples.dot(weights)
