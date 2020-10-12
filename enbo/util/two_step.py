#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Batch Knowledge Gradient (KG) via one-shot optimization as introduced in
[Balandat2019botorch]_. For broader discussion of KG see also
[Frazier2008knowledge]_, [Wu2016parallelkg]_.

.. [Balandat2019botorch]
    M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson,
    and E. Bakshy. BoTorch: Programmable Bayesian Optimziation in PyTorch.
    ArXiv 2019.

.. [Frazier2008knowledge]
    P. Frazier, W. Powell, and S. Dayanik. A Knowledge-Gradient policy for
    sequential information collection. SIAM Journal on Control and Optimization,
    2008.

.. [Wu2016parallelkg]
    J. Wu and P. Frazier. The parallel knowledge gradient method for batch
    bayesian optimization. NIPS 2016.
"""

from copy import deepcopy
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from botorch import settings
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    qSimpleRegret,
    qExpectedImprovement,
)
from botorch.acquisition.objective import (
    AcquisitionObjective,
    MCAcquisitionObjective,
    ScalarizedObjective,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from util.gauss_hermite import GaussHermiteSampler


class TwoStep(qKnowledgeGradient):
    r"""Batch two-step lookahead using one-shot optimization.

    This computes the batch Knowledge Gradient using fantasies for the outer
    expectation and either the model posterior mean or MC-sampling for the inner
    expectation.

    In addition to the design variables, the input `X` also includes variables
    for the optimal designs for each of the fantasy models. For a fixed number
    of fantasies, all parts of `X` can be optimized in a "one-shot" fashion.
    """

    def __init__(
        self,
        model: Model,
        num_fantasies: Optional[int] = 64,
        sampler: Optional[MCSampler] = None,
        objective: Optional[AcquisitionObjective] = None,
        inner_sampler: Optional[MCSampler] = None,
        X_pending: Optional[Tensor] = None,
        current_value: Optional[Tensor] = None,
        q1: Optional[int] = 1,
    ) -> None:
        super().__init__(
            model,
            num_fantasies,
            sampler,
            objective,
            inner_sampler,
            X_pending,
            current_value,
        )
        self.q1 = q1

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X`.

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """
        if self.q1 == 0:
            X_actual = X
        else:
            X_actual, X_fantasies = _split_fantasy_points(
                X=X, n_f=self.num_fantasies, q1=self.q1
            )

        current_value = (
            self.current_value if self.current_value else self.model.train_targets.max()
        )
        ei = ExpectedImprovement(model=self.model, best_f=current_value)
        one_step_utility = ei(X_actual)

        if self.q1 == 0:
            return one_step_utility

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual,
            sampler=self.sampler,
            observation_noise=True,  # noise=self.model.likelihood.noise_covar.noise
        )
        # get the value function
        # value_function = _get_value_function(
        #     model=fantasy_model, objective=self.objective, sampler=self.inner_sampler
        # )
        # fantasy_model.train_targets: num_fantasies x batch_size x (num_actual_train + num_fantasy_train)
        best_f = fantasy_model.train_targets.max(dim=-1)[0]
        if self.q1 == 1:  # two-step, use analytic EI
            value_function = ExpectedImprovement(model=fantasy_model, best_f=best_f)
        else:
            sampler = SobolQMCNormalSampler(10000)
            value_function = qExpectedImprovementBatch(
                model=fantasy_model, sampler=sampler, best_f=best_f.unsqueeze(-1)
            )

        # make sure to propagate gradients to the fantasy model train inputs
        with settings.propagate_grads(True):
            values = value_function(X=X_fantasies)  # num_fantasies x batch_size x q x d

        # return average over the fantasy samples
        if isinstance(self.sampler, GaussHermiteSampler):
            weighted_values = values * self.sampler.base_weights
            future_utility = torch.sum(weighted_values, dim=0)
        else:
            future_utility = values.mean(dim=0)

        return one_step_utility + future_utility

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimzation.

        Args:
            q: The number of candidates to consider jointly.

        Returns:
            The augmented size for one-shot optimzation (including variables
            parameterizing the fantasy solutions).
        """
        return q + self.num_fantasies * self.q1

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.

        Args:
            X_full: A `b x (q + num_fantasies) x d`-dim Tensor with `b`
                t-batches of `q + num_fantasies` design points each.

        Returns:
            A `b x q x d`-dim Tensor with `b` t-batches of `q` design points each.
        """
        return X_full[..., : -self.num_fantasies * self.q1, :]


class DummyKG(MCAcquisitionFunction):
    def __init__(self, qKG, X_actual):
        super().__init__(model=qKG.model)
        self.qKG = qKG
        self.X_actual = X_actual

    def forward(self, X):  # X is fantasized points
        # X: batch_shape x (num_fantasies*q1) x d -> batch_shape x num_fantasies x q1 x d
        # X = _reshape_fantasy_points(X, self.qKG.num_fantasies, self.qKG.q1)

        X_actual = self.X_actual.repeat(*(X.shape[:-2]), 1, 1)
        res = self.qKG(torch.cat([X_actual, X], dim=-2))
        return res


class EvalKG(MCAcquisitionFunction):
    def __init__(self, qKG, bounds):
        super().__init__(model=qKG.model)
        self.qKG = qKG
        self.bounds = bounds

    def forward(self, X):  # X is an actual candidate

        dKG = DummyKG(self.qKG, X)
        if hasattr(self.qKG, "q1"):
            q = self.qKG.num_fantasies * self.qKG.q1
        else:
            q = self.qKG.num_fantasies
        _, val = optimize_acqf(dKG, self.bounds, q=q, num_restarts=20, raw_samples=512)
        return val


def _split_fantasy_points(X: Tensor, n_f: int, q1: int = 1) -> Tuple[Tensor, Tensor]:
    r"""Split a one-shot optimization input into actual and fantasy points

    Args:
        X: A `batch_shape x (q + n_f*q1) x d`-dim tensor of actual and fantasy
            points

    Returns:
        2-element tuple containing

        - A `batch_shape x q x d`-dim tensor `X_actual` of input candidates.
        - A `n_f x batch_shape x q1 x d`-dim tensor `X_fantasies` of fantasy
            points, where `X_fantasies[i, batch_idx]` is the i-th fantasy batch of points
            associated with the batch indexed by `batch_idx`.
    """
    if n_f > X.size(-2):
        raise ValueError(
            f"n_f ({n_f}) must be less than the q-batch dimension of X ({X.size(-2)})"
        )
    split_sizes = [X.size(-2) - n_f * q1, n_f * q1]
    X_actual, X_fantasies = torch.split(X, split_sizes, dim=-2)
    # X_fantasies is b x (num_fantasies*q1) x d, needs to be num_fantasies x b x q1 x d
    # for batch mode evaluation with batch shape num_fantasies x b.

    # num_fantasies x b x q1 x d
    X_fantasies = _reshape_fantasy_points(X_fantasies, n_f, q1)

    return X_actual, X_fantasies


def _reshape_fantasy_points(X: Tensor, n_f: int, q1: int) -> Tensor:
    (b, num_fantasies, d) = X.shape
    assert (
        num_fantasies == n_f * q1
    ), f"X.shape[-1] {X.shape[-1]} should equal n_f*q1 ({n_f}*{q1})"
    # b x (num_fantasies*q1) x d --> b x num_fantasies x q1 x d
    X = X.view(b, n_f, q1, d)
    # b x num_fantasies x q1 x d --> num_fantasies x b x q1 x d
    X = X.permute(-3, *range(X.dim() - 3), -2, -1)
    return X


class qExpectedImprovementBatch(qExpectedImprovement):
    r"""MC-based batch Expected Improvement.

    This is a modification of qExpectedImprovement to allow a batch of best_f values

    This computes qEI by
    (1) sampling the joint posterior over q points
    (2) evaluating the improvement over the current best for each sample
    (3) maximizing over q
    (4) averaging over the samples

    `qEI(X) = E(max(max Y - best_f, 0)), Y ~ f(X), where X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1000)
        >>> qEI = qExpectedImprovement(model, best_f, sampler)
        >>> qei = qEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Expected Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless).
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            X_pending:  A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.  Concatenated into X upon
                forward call.  Copied and set to have no gradient.
        """
        super(qExpectedImprovement, self).__init__(
            model=model, sampler=sampler, objective=objective, X_pending=X_pending
        )
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(float(best_f))

        self.register_buffer("best_f", best_f)
