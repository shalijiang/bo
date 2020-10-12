#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import Size, Tensor
from torch.distributions import Beta
from torch.nn import ModuleList

from ..exceptions.errors import UnsupportedError
from ..exceptions.warnings import BotorchWarning
from ..models.model import Model
from ..optim.initializers import initialize_q_batch
from ..optim.optimize import optimize_acqf
from ..sampling.samplers import MCSampler, SobolQMCNormalSampler
from ..utils.transforms import match_batch_shape, unnormalize
from .acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from .analytic import AnalyticAcquisitionFunction, PosteriorMean
from .monte_carlo import MCAcquisitionFunction, qExpectedImprovement
from .objective import AcquisitionObjective, MCAcquisitionObjective, ScalarizedObjective


TAcqfArgConstructor = Callable[[Model, Tensor], Dict[str, Any]]


TAcqfArgConstructor = Callable[[Model, Tensor], Dict[str, Any]]


def make_best_f(model: Model, X: Tensor) -> Dict[str, Any]:
    r"""Extract the best observed training input from the model."""
    return {"best_f": model.train_targets.max(dim=-1).values}


class qMultiStepLookahead(MCAcquisitionFunction, OneShotAcquisitionFunction):
    r"""MC-based batch Multi-Step Look-Ahead (one-shot optimization)."""

    def __init__(
        self,
        model: Model,
        batch_sizes: List[int],
        num_fantasies: List[int] = None,
        samplers: List[MCSampler] = None,
        valfunc_cls: Optional[List[Optional[Type[AcquisitionFunction]]]] = None,
        valfunc_argfacs: Optional[List[Optional[TAcqfArgConstructor]]] = None,
        objective: Optional[AcquisitionObjective] = None,
        inner_mc_samples: Optional[List[int]] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Multi-Step Look-Ahead (one-shot optimization).
        Performs a `k`-step lookahead by means of repeated fantasizing.
        Allows to specify the stage value functions by passing the respective class
        objects via the `valfunc_cls` list. Optionally, `valfunc_argfacs` takes a list
        of callables that generate additional kwargs for these constructors. By default,
        `valfunc_cls` will be chosen as `[None, ..., None, PosteriorMean]`, which
        corresponds to the (parallel) multi-step KnowledgeGradient. If, in addition,
        `k=1` and `q_1 = 1`, this reduces to the classic Knowledge Gradient.
        WARNING: The complexity of evaluating this function is exponential in the number
        of lookahead steps!
        Args:
            model: A fitted model.
            batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes for the
                `k` look-ahead steps.
            num_fantasies: A list `[f_1, ..., f_k]` containing the number of fantasy
                points to use for the `k` look-ahead steps.
            valfunc_cls: A list of `k + 1` acquisition function classes to be used as
                the (stage + terminal) value functions. Each element (except for the
                last one) can be `None`, in which case a zero stage value is assumed for
                the respective stage. If `None`, this defaults to
                `[None, ..., None, PosteriorMean]`
            valfunc_argfacs: A list of `k + 1` "argument factories", i.e. callables that
                map a `Model` and input tensor `X` to a dictionary of kwargs for the
                respective stage value function constructor (e.g. `best_f` for
                `ExpectedImprovement`). If None, only the standard (`model`, `sampler`
                and `objective`) kwargs will be used.
            objective: The objective under which the output is evaluated. If `None`, use
                the model output (requires a single-output model). If a
                `ScalarizedObjective` and `value_function_cls` is a subclass of
                `AnalyticAcquisitonFunction`, then the analytic posterior mean is used.
                Otherwise the objective is MC-evaluated (using `inner_sampler`).
            inner_mc_samples: A list `[n_0, ..., n_k]` containing the number of MC
                samples to be used for evaluating the stage value function. Ignored if
                the objective is `None` or a `ScalarizedObjective`.
            X_pending: A `m x d`-dim Tensor of `m` design points that have points that
                have been submitted for function evaluation but have not yet been
                evaluated. Concatenated into `X` upon forward call. Copied and set to
                have no gradient.
        """
        super(MCAcquisitionFunction, self).__init__(model=model)
        self.batch_sizes = batch_sizes
        assert num_fantasies or samplers, "num_fantasies and samplers are both None!"
        if samplers:  # override num_fantasies
            num_fantasies = [sampler.sample_shape[0] for sampler in samplers]
        else:
            # construct samplers for the look-ahead steps
            samplers: List[MCSampler] = [
                SobolQMCNormalSampler(
                    num_samples=nf, resample=False, collapse_batch_dims=True
                )
                for nf in num_fantasies
            ]
        self.num_fantasies = num_fantasies
        # By default do not use stage values and use PosteriorMean as terminal value
        # function (= multi-step KG)
        if valfunc_cls is None:
            valfunc_cls = [None for _ in num_fantasies] + [PosteriorMean]
        if inner_mc_samples is None:
            inner_mc_samples = [None] * (1 + len(num_fantasies))
        # TODO: Allow passing in inner samplers directly
        inner_samplers = _construct_inner_samplers(
            batch_sizes=batch_sizes,
            valfunc_cls=valfunc_cls,
            objective=objective,
            inner_mc_samples=inner_mc_samples,
        )
        if valfunc_argfacs is None:
            valfunc_argfacs = [None] * (1 + len(batch_sizes))
        self.objective = objective
        self.set_X_pending(X_pending)
        self.samplers = ModuleList(samplers)
        self.inner_samplers = ModuleList(inner_samplers)
        self._valfunc_cls = valfunc_cls
        self._valfunc_argfacs = valfunc_argfacs

    def forward(self, X: Union[Tensor, List[Tensor]]) -> Tensor:
        r"""Evaluate qMultiStepLookahead on the candidate set X.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i` is respective number of fantasies.

        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        if not isinstance(X, list):
            batch_shape, shapes, sizes = self.get_split_shapes(X=X)
            # # Each X_i in Xsplit has shape batch_shape x qtilde x d with
            # # qtilde = f_i * ... * f_1 * q_i
            Xsplit = torch.split(X, sizes, dim=-2)
            # now reshape (need to permute batch_shape and qtilde dimensions for i > 0)
            perm = [-2] + list(range(len(batch_shape))) + [-1]
            X0 = Xsplit[0].reshape(shapes[0])
            Xother = [
                X.permute(*perm).reshape(shape)
                for X, shape in zip(Xsplit[1:], shapes[1:])
            ]
            # concatenate in pending points
            if self.X_pending is not None:
                X0 = torch.cat([X0, match_batch_shape(self.X_pending, X0)], dim=-2)
            X_list = [X0] + Xother
        else:
            X_list = X

        return _step(
            model=self.model,
            Xs=X_list,
            samplers=self.samplers,
            valfunc_cls=self._valfunc_cls,
            valfunc_argfacs=self._valfunc_argfacs,
            inner_samplers=self.inner_samplers,
            objective=self.objective,
            running_val=None,
        )

    @property
    def _num_auxiliary(self) -> int:
        r"""Number of auxiliary variables in the q-batch dimension.
        Returns:
             `q_aux` s.t. `q + q_aux = augmented_q_batch_size`
        """
        return np.dot(self.batch_sizes, np.cumprod(self.num_fantasies)).item()

    def get_augmented_q_batch_size(self, q: int) -> int:
        r"""Get augmented q batch size for one-shot optimzation.
        Args:
            q: The number of candidates to consider jointly.
        Returns:
            The augmented size for one-shot optimzation (including variables
            parameterizing the fantasy solutions): `q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`
        """
        return q + self._num_auxiliary

    def get_split_shapes(self, X: Tensor) -> Tuple[Size, List[Size], List[int]]:
        r"""
        """
        batch_shape, (q_aug, d) = X.shape[:-2], X.shape[-2:]
        q = q_aug - self._num_auxiliary
        batch_sizes = [q] + self.batch_sizes
        # X_i needs to have shape f_i x .... x f_1 x batch_shape x q_i x d
        shapes = [
            torch.Size(self.num_fantasies[:i][::-1] + [*batch_shape, q_i, d])
            for i, q_i in enumerate(batch_sizes)
        ]
        # Each X_i in Xsplit has shape batch_shape x qtilde x d with
        # qtilde = f_i * ... * f_1 * q_i
        sizes = [s[:-3].numel() * s[-2] for s in shapes]
        return batch_shape, shapes, sizes

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        r"""We only return X as the set of candidates post-optimization.
        Args:
            X_full: A `batch_shape x q' x d`-dim Tensor with `q'` design points for
                each batch, where `q' = q + f_1 q_1 + f_2 f_1 q_2 + ...`.
        Returns:
            A `batch_shape x q x d`-dim Tensor with `q` design points for each batch.
        """
        return X_full[..., : -self._num_auxiliary, :]


def _step(
    model: Model,
    Xs: List[Tensor],
    samplers: List[Optional[MCSampler]],
    valfunc_cls: List[Optional[Type[AcquisitionFunction]]],
    valfunc_argfacs: List[Optional[TAcqfArgConstructor]],
    inner_samplers: List[Optional[MCSampler]],
    objective: AcquisitionObjective,
    running_val: Optional[Tensor] = None,
    first_step: bool = False,
    sample_weights: Optional[
        List[Tensor]
    ] = None,  # can't use [] as default, very weird
) -> Tensor:
    r"""Recursive multi-step look-ahead computation.
    Helper function computing the "value-to-go" of a multi-step lookahead scheme.
    Args:
        model: A Model of appropriate batch size. Specifically, it must be possible to
            evaluate the model's posterior at `Xs[0]`.
        Xs: A list `[X_j, ..., X_k]` of tensors, where `X_i` has shape
            `f_i x .... x f_1 x batch_shape x q_i x d`.
        samplers: A list of `k - j` samplers, such that the number of samples of sampler
            `i` is `f_i`. The last element of this list is considered the
            "inner sampler", which is used for evaluating the objective in case it is an
            MCAcquisitionObjective.
        valfunc_cls: A list of acquisition function class to be used as the (stage +
            terminal) value functions. Each element (except for the last one) can be
            `None`, in which case a zero stage value is assumed for the respective
            stage.
        valfunc_argfacs: A list of callables that map a `Model` and input tensor `X` to
            a dictionary of kwargs for the respective stage value function constructor.
            If `None`, only the standard `model`, `sampler` and `objective` kwargs will
            be used.
        inner_samplers: A list of `MCSampler` objects, each to be used in the stage
            value function at the corresponding index.
        objective: The AcquisitionObjective under which the model output is evaluated.
        running_val: As `batch_shape`-dim tensor containing the current running value.
        first_step: If True, this is considered to be the first step (resulting
            in not propagating gradients through the training inputs of the model).
    Returns:
        A `b`-dim tensor containing the multi-step value of the design `X`.
    """
    if len(Xs) != len(samplers) + 1:
        raise ValueError("Must have as many samplers as look-ahead steps")

    # compute stage value
    StageValFuncClass = valfunc_cls[0]
    if StageValFuncClass is not None:
        common_kwargs: Dict[str, Any] = {"model": model, "objective": objective}
        if issubclass(StageValFuncClass, MCAcquisitionFunction):
            common_kwargs["sampler"] = inner_samplers[0]
        arg_fac = valfunc_argfacs[0]
        kwargs = arg_fac(model=model, X=Xs[0]) if arg_fac is not None else {}
        stage_val_func = StageValFuncClass(**common_kwargs, **kwargs)
        stage_val = stage_val_func(X=Xs[0])
        # shape of stage_val is (inner_mc_samples) x f_k x ... x f_1 x batch_shape
        # we average across all dimensions except for the batch dimension
        if sample_weights:
            for i, sample_weight in enumerate(sample_weights[::-1]):
                nf = len(sample_weight)
                extend_dims = [1] * (len(stage_val.shape) - 1)
                # stage_val:     f_k x ... x f_1
                # sample_weight: f_k x 1 x ... x 1
                stage_val = stage_val * sample_weight.view(
                    nf, *extend_dims
                )  # elementwise multiplication (broadcast)
                stage_val = stage_val.sum(dim=0)
        else:
            stage_val = stage_val.view(-1, stage_val.size(-1)).mean(dim=0)

        # update running value
        running_val = stage_val if running_val is None else running_val + stage_val

    # base case: no more fantasizing, return value
    if len(Xs) == 1:
        return running_val

    # construct fantasy model (with batch shape f_{j+1} x ... x f_1 x batch_shape)
    fantasy_model = model.fantasize(
        X=Xs[0],
        sampler=samplers[0],
        observation_noise=True,
        propagate_grads=not first_step,
    )
    if hasattr(samplers[0], "base_weights"):
        sample_weights = sample_weights or []
        sample_weights.append(samplers[0].base_weights)
    return _step(
        model=fantasy_model,
        Xs=Xs[1:],
        samplers=samplers[1:],
        valfunc_cls=valfunc_cls[1:],
        valfunc_argfacs=valfunc_argfacs[1:],
        inner_samplers=inner_samplers[1:],
        objective=objective,
        running_val=running_val,
        sample_weights=sample_weights,
    )


def _construct_inner_samplers(
    batch_sizes: List[int],
    valfunc_cls: List[Optional[Type[AcquisitionFunction]]],
    inner_mc_samples: List[Optional[int]],
    objective: Optional[AcquisitionObjective] = None,
) -> List[Optional[MCSampler]]:
    r"""Check validity of inputs and construct inner samplers.
    Helper function to be used internally for constructing inner samplers.
    Args:
        batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes for the
            `k` look-ahead steps.
        valfunc_cls: A list of `k + 1` acquisition function classes to be used as the
            (stage + terminal) value functions. Each element (except for the last one)
            can be `None`, in which case a zero stage value is assumed for the
            respective stage.
        inner_mc_samples: A list `[n_0, ..., n_k]` containing the number of MC
            samples to be used for evaluating the stage value function. Ignored if
            the objective is `None` or a `ScalarizedObjective`.
        objective: The objective under which the output is evaluated. If `None`, use
            the model output (requires a single-output model). If a
            `ScalarizedObjective` and `value_function_cls` is a subclass of
            `AnalyticAcquisitonFunction`, then the analytic posterior mean is used.
            Otherwise the objective is MC-evaluated (using `inner_sampler`).
    Returns:
        A list with `k + 1` elements that are either `MCSampler`s or `None.
    """
    inner_samplers = []
    for q, vfc, mcs in zip([None] + batch_sizes, valfunc_cls, inner_mc_samples):
        if vfc is None:
            inner_samplers.append(None)
        elif vfc == qMultiStepLookahead:
            raise UnsupportedError(
                "qMultiStepLookahead not supported as a value function "
                "(I see what you did there, nice try...)."
            )
        elif issubclass(vfc, AnalyticAcquisitionFunction):
            if objective is not None and not isinstance(objective, ScalarizedObjective):
                raise UnsupportedError(
                    "Only objectives of type ScalarizedObjective are supported "
                    "for analytic value functions."
                )
            # At this point, we don't know the initial q-batch size here
            if q is not None and q > 1:
                raise UnsupportedError(
                    "Only batch sizes of q=1 are supported for analytic value "
                    "functions."
                )
            if q is not None and mcs is not None:
                warnings.warn(
                    "inner_mc_samples is ignored for analytic acquistion functions",
                    BotorchWarning,
                )
            inner_samplers.append(None)
        else:
            if objective is not None and not isinstance(
                objective, MCAcquisitionObjective
            ):
                raise UnsupportedError(
                    "Only objectives of type MCAcquisitionObjective are supported "
                    "for MC value functions."
                )
            if mcs is None:
                # TODO: Default values for mc samples
                raise NotImplementedError
            inner_sampler = SobolQMCNormalSampler(
                num_samples=mcs, resample=False, collapse_batch_dims=True
            )
            inner_samplers.append(inner_sampler)
    return inner_samplers


def warmerstart_multistep(
    acq_function: qMultiStepLookahead,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    last_iter_info: Dict,
) -> Tensor:
    r"""Warm-start initialization for multi-step look-ahead acquisition functions.
    For now uses the same q as in `full_optimizer`. TODO: allow different values of `q`
    Args:
        acq_function: A qMultiStepLookahead acquisition function.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of features.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        full_optiizer: The full tree of optimizers of the previous iteration. Typically
            obtained by passing `return_best_only=False` and `return_full_tree=True`
            into `optimize_acqf`.
        kwargs: Optimization kwargs.
    This is a very simple initialization heuristic.
    TODO: Use the observed values to identify the fantasy sub-tree that is closest to
    the observed value.
    """
    # new_X = _batch_initializer(acq_function, bounds, num_restarts)

    new_X = _subtree_initializer(acq_function, last_iter_info)
    if new_X is None:
        return None
    new_X = new_X.repeat(num_restarts, 1, 1)
    perturb = 0.1
    new_X[1:] = new_X[1:] + perturb * torch.randn_like(new_X[1:]) * new_X[1:]
    return new_X


def _batch_initializer(
    acq_function: qMultiStepLookahead, bounds: Tensor, num_restarts: int
):
    horizon = len(acq_function.samplers) + 1
    qmc_sampler = SobolQMCNormalSampler(num_samples=512)
    model = acq_function.model
    best_f = torch.max(model.train_targets)
    qEI = qExpectedImprovement(model=model, best_f=best_f, sampler=qmc_sampler)
    qei_num_restarts = int(np.ceil(num_restarts / horizon))
    batch_optimizer, _ = optimize_acqf(
        acq_function=qEI,
        bounds=bounds,
        q=horizon,
        num_restarts=qei_num_restarts,
        raw_samples=512,
        return_best_only=False
    )
    # batch_optimizer: qei_num_restarts x horizon x d
    tree_roots = batch_optimizer.reshape(qei_num_restarts * horizon, -1)[:num_restarts]
    # construct fantasy model (with batch shape f_{j+1} x ... x f_1 x batch_shape)
    fantasy_model = model.fantasize(
        X=tree_roots,
        sampler=acq_function.samplers[0],
        observation_noise=True,
        propagate_grads=False,
    )

    # best_f = fantasy_model.train_targets.max(dim=-1).values
    qEI1 = qExpectedImprovement(model=fantasy_model, best_f=best_f, sampler=qmc_sampler)
    qei_num_restarts1 = int(np.ceil(num_restarts / horizon))
    batch_optimizer1, _ = optimize_acqf(
        acq_function=qEI1,
        bounds=bounds,
        q=horizon-1,
        num_restarts=2,
        raw_samples=512,
        return_best_only=True,
    )





def _subtree_initializer(acq_function: qMultiStepLookahead, last_iter_info: Dict):
    """
    Use the full tree optimizer from last iteration,
    and extract the subtree with fantasy y value closest to the actual observed value.
    The last layer is simple random perturbation of the last layer from last iteration full optimizer

    Args:
        acq_function: the qMultiStepLookahead acquisition function
        last_iter_info: a dictionary containing info from last iteration
            -full_optimizer: Tensor of shape num_restarts x q' x d
            -observed_y: actual observed function value
            -best_restart_idx: index of the best restart
    Returns:
        new_X: Tensor of shape q' x d, the extracted subtree
    """
    full_optimizer = last_iter_info.get("full_optimizer", None)

    if full_optimizer is None:
        return None
    num_fantasies = [sampler.sample_shape for sampler in acq_function.samplers]
    if not all([n == num_fantasies[0] for n in num_fantasies[1:]]):
        warnings.warn(
            "require num_fantasies be same for each stage, fall back to default initialization"
        )
        return None

    best_restart_idx = last_iter_info.get("best_restart_idx", None)
    observed_y = last_iter_info.get("obversed_y", None)
    standardize_y = last_iter_info.get("stardardize_y", None)
    if standardize_y is not None:
        mu, sig = standardize_y
        observed_y = (observed_y - mu) / sig
    batch_shape, shapes, sizes = acq_function.get_split_shapes(full_optimizer)
    Xopts = torch.split(full_optimizer, sizes, dim=-2)
    perm = [-2] + list(range(len(batch_shape))) + [-1]
    X0 = Xopts[0].reshape(shapes[0])
    Xother = [
        X.permute(*perm).reshape(shape) for X, shape in zip(Xopts[1:], shapes[1:])
    ]

    with torch.no_grad():
        fantasy_model = acq_function.model.fantasize(
            X0[best_restart_idx],
            sampler=acq_function.samplers[0],  # re-using the sampler is critical
        )

    fantasy_samples = fantasy_model.train_targets[
        ..., acq_function.model.train_targets.size(-1) :
    ]

    chosen_branch_fantvals = fantasy_samples.squeeze()

    closest_fantasy_idx = torch.argmin(torch.abs(chosen_branch_fantvals - observed_y))
    new_X0 = Xother[0][closest_fantasy_idx][best_restart_idx]
    new_Xother = [
        x[..., closest_fantasy_idx, best_restart_idx, :, :] for x in Xother[1:]
    ]

    X_final = Xother[-1][..., best_restart_idx, :, :]
    # the following two lines are copied from Max's notebook
    eta = 0.2
    X_final = X_final + eta * (torch.exp(-torch.rand_like(X_final)) - X_final)
    new_Xother.append(X_final)

    # transform the shape back
    new_Xopts = [new_X0.reshape(X0[best_restart_idx].shape)]
    for X, X_orig in zip(new_Xother, Xopts[1:]):
        perm = [-3] + list(range(len(X.shape[:-3]))) + [-2, -1]
        X = X.permute(*perm).reshape(X_orig[best_restart_idx].shape)
        new_Xopts.append(X)
    new_X = torch.cat(new_Xopts)

    return new_X


def warmstart_multistep(
    acq_function: qMultiStepLookahead,
    bounds: Tensor,
    num_restarts: int,
    raw_samples: int,
    full_optimizer: Tensor,
    **kwargs: Any,
) -> Tensor:
    r"""Warm-start initialization for multi-step look-ahead acquisition functions.
    For now uses the same q as in `full_optimizer`. TODO: allow different values of `q`
    Args:
        acq_function: A qMultiStepLookahead acquisition function.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of features.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        full_optiizer: The full tree of optimizers of the previous iteration. Typically
            obtained by passing `return_best_only=False` and `return_full_tree=True`
            into `optimize_acqf`.
        kwargs: Optimization kwargs.
    This is a very simple initialization heuristic.
    TODO: Use the observed values to identify the fantasy sub-tree that is closest to
    the observed value.
    """
    batch_shape, shapes, sizes = acq_function.get_split_shapes(full_optimizer)
    Xopts = torch.split(full_optimizer, sizes, dim=-2)
    tkwargs = {"device": Xopts[0].device, "dtype": Xopts[0].dtype}

    B = Beta(torch.ones(1, **tkwargs), 3 * torch.ones(1, **tkwargs))

    def mixin_layer(X: Tensor, bounds: Tensor, eta: float) -> Tensor:
        perturbations = unnormalize(B.sample(X.shape).squeeze(-1), bounds)
        return (1 - eta) * X + eta * perturbations

    def make_init_tree(Xopts: List[Tensor], bounds: Tensor, etas: Tensor) -> Tensor:
        Xtrs = [mixin_layer(X=X, bounds=bounds, eta=eta) for eta, X in zip(etas, Xopts)]
        return torch.cat(Xtrs, dim=-2)

    def mixin_tree(T: Tensor, bounds: Tensor, alpha: float) -> Tensor:
        return (1 - alpha) * T + alpha * unnormalize(torch.rand_like(T), bounds)

    n_repeat = math.ceil(raw_samples / batch_shape[0])
    alphas = torch.linspace(0, 0.75, n_repeat, **tkwargs)
    etas = torch.linspace(0.1, 1.0, len(Xopts), **tkwargs)

    X_full = torch.cat(
        [
            mixin_tree(
                T=make_init_tree(Xopts=Xopts, bounds=bounds, etas=etas),
                bounds=bounds,
                alpha=alpha,
            )
            for alpha in alphas
        ],
        dim=0,
    )

    with torch.no_grad():
        Y_full = acq_function(X_full)
    X_init = initialize_q_batch(X=X_full, Y=Y_full, n=num_restarts, eta=1.0)
    return X_init[:raw_samples]
