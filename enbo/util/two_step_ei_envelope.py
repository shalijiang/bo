import torch
from botorch.models.model import Model
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    AnalyticAcquisitionFunction,
)
from botorch.acquisition.monte_carlo import qExpectedImprovement
from torch import Tensor
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.gen import gen_candidates_torch
from botorch.sampling.samplers import (
    MCSampler,
    GaussHermiteSampler,
    SobolQMCNormalSampler,
)
from typing import Union, Optional, Dict
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition import ScalarizedObjective
from botorch.acquisition.two_step_ei import qExpectedImprovementBatch
from botorch import settings
from botorch.utils.sampling import draw_sobol_samples


class TwoStepEIEnvelope(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        bounds: Tensor,
        sampler: Optional[MCSampler] = None,
        inner_sampler: Optional[MCSampler] = None,
        q1: Optional[int] = 1,
        options: Optional[Dict[str, Union[bool, float, int, str]]] = {},
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = True,
    ) -> None:
        r"""Two-step Expected Improvement based on:
        Wu & Frazier, Practical Two-Step Lookahead Bayesian Optimization, NeurIPS 2019.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            bounds: a 2 x d tensor specifying the range of each dimension
            sampler: used to sample y from its posterior
            q1: int, batch size of the second step
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)
        self.bounds = bounds
        self.sampler = sampler
        self.inner_sampler = inner_sampler or SobolQMCNormalSampler(num_samples=512)
        self.q1 = q1
        self.options = options

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Two-Step EI on the candidate set X.

        Args:
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `b1 x ... bk`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        acq_func = ExpectedImprovement(model=self.model, best_f=self.best_f)
        immediate_utility = acq_func(X)
        if self.q1 == 0:
            return immediate_utility

        if X.dim() < 3:
            X = X.unsqueeze(0)

        batch_size, q, d = X.shape

        fantasy_model = self.model.fantasize(
            X=X, sampler=self.sampler, observation_noise=True
        )
        best_f = fantasy_model.train_targets.max(dim=-1)[0]
        # assume _sample_shape=torch.Size([num_samples])
        num_fantasies = self.sampler.sample_shape[0]
        with torch.enable_grad():
            if self.q1 == 1:  # two-step, use analytic EI
                value_function = ExpectedImprovement(model=fantasy_model, best_f=best_f)
            else:
                value_function = qExpectedImprovement(
                    model=fantasy_model, sampler=self.inner_sampler, best_f=best_f
                )

            def joint_value_function(X):

                # X reshape to batch_shape x fantasies x batch_size x q x d
                batch_size_joint = X.shape[0]
                X_prime = X.view(
                    batch_size_joint, num_fantasies, batch_size, self.q1, d
                )
                # values: batch_size_joint x num_fantasies x batch_size
                values = value_function(X_prime)
                return values.sum(tuple(range(1, len(values.shape))))

            joint_optim_size = num_fantasies * batch_size * self.q1
            # can tune num_restarts, raw_samples, and maxiter to tradeoff efficiency and accuracy
            num_restarts = 20
            seed = self.options.get("seed", 0)
            method = self.options.get("method", "scipy")
            if method == "scipy":  # by default L-BFGS-B is used
                num_batches = self.options.get("num_batches", 5)
                X_fantasies, _ = optimize_acqf(
                    acq_function=joint_value_function,
                    bounds=self.bounds,
                    q=joint_optim_size,
                    num_restarts=num_restarts,
                    raw_samples=512,
                    options={
                        "maxiter": 500,
                        "seed": seed,
                        "batch_limit": round(num_restarts / num_batches),
                    },
                )
            elif method == "torch" or method == "sgd":  # by default Adam is used
                bounds = self.bounds
                Xinit = gen_batch_initial_conditions(
                    acq_function=joint_value_function,
                    bounds=bounds,
                    q=joint_optim_size,
                    num_restarts=50,
                    raw_samples=1000,
                    options={
                        "nonnegative": True,
                        "seed": self.options.get("seed", None),
                    },
                )
                # Xinit = draw_sobol_samples(bounds=bounds, n=100, q=joint_optim_size, seed=self.options.get("seed", None))
                optimizer = torch.optim.SGD if method == "sgd" else torch.optim.Adam
                batch_candidates, batch_acq_values = gen_candidates_torch(
                    initial_conditions=Xinit,
                    acquisition_function=joint_value_function,
                    lower_bounds=bounds[0],
                    upper_bounds=bounds[1],
                    optimizer=optimizer,
                    options={
                        "maxiter": 500,
                        "lr": 1.0,
                        "scheduler_on": True,
                        "gamma": 0.7,
                    },
                    # options={"maxiter": 300},
                    verbose=False,
                )
                best = torch.argmax(batch_acq_values.view(-1), dim=0)
                X_fantasies = batch_candidates[best].detach()

        X_fantasies = X_fantasies.view(num_fantasies, batch_size, self.q1, d)
        with settings.propagate_grads(True):
            values = value_function(X_fantasies)

        if isinstance(self.sampler, GaussHermiteSampler):
            weighted_values = values * self.sampler.base_weights
            future_utility = torch.sum(weighted_values, dim=0)
        else:
            future_utility = values.mean(dim=0)

        return immediate_utility + future_utility
