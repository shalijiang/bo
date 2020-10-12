#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for model fitting.
"""

import logging
import warnings
from copy import deepcopy
from typing import Any, Callable

from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from .exceptions.errors import UnsupportedError
from .exceptions.warnings import BotorchWarning, OptimizationWarning
from .models.converter import batched_to_model_list, model_list_to_batched
from .models.gpytorch import BatchedMultiOutputGPyTorchModel
from .optim.fit import fit_gpytorch_scipy
from .optim.utils import sample_all_priors


FAILED_CONVERSION_MSG = (
    "Failed to convert ModelList to batched model. "
    "Performing joint instead of sequential fitting."
)


def fit_gpytorch_model(
    mll: MarginalLogLikelihood, optimizer: Callable = fit_gpytorch_scipy, **kwargs: Any
) -> MarginalLogLikelihood:
    r"""Fit hyperparameters of a GPyTorch model.

    On optimizer failures, a new initial condition is sampled from the
    hyperparameter priors and optimization is retried. The maximum number of
    retries can be passed in as a `max_retries` kwarg (default is 5).

    Optimizer functions are in botorch.optim.fit.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        optimizer: The optimizer function.
        kwargs: Arguments passed along to the optimizer function, including
            `max_retries` and `sequential` (controls the fitting of `ModelListGP`
            and `BatchedMultiOutputGPyTorchModel` models) or `approx_mll`
            (whether to use gpytorch's approximate MLL computation).

    Returns:
        MarginalLogLikelihood with optimized parameters.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> fit_gpytorch_model(mll)
    """
    sequential = kwargs.pop("sequential", True)
    max_retries = kwargs.pop("max_retries", 5)
    if isinstance(mll, SumMarginalLogLikelihood) and sequential:
        for mll_ in mll.mlls:
            fit_gpytorch_model(
                mll=mll_, optimizer=optimizer, max_retries=max_retries, **kwargs
            )
        return mll
    elif (
        isinstance(mll.model, BatchedMultiOutputGPyTorchModel)
        and mll.model._num_outputs > 1
        and sequential
    ):
        try:  # check if backwards-conversion is possible
            model_list = batched_to_model_list(mll.model)
            model_ = model_list_to_batched(model_list)
            mll_ = SumMarginalLogLikelihood(model_list.likelihood, model_list)
            fit_gpytorch_model(
                mll=mll_,
                optimizer=optimizer,
                sequential=True,
                max_retries=max_retries,
                **kwargs,
            )
            model_ = model_list_to_batched(mll_.model)
            mll.model.load_state_dict(model_.state_dict())
            return mll.eval()
        except (NotImplementedError, UnsupportedError, RuntimeError, AttributeError):
            warnings.warn(FAILED_CONVERSION_MSG, BotorchWarning)
            return fit_gpytorch_model(
                mll=mll, optimizer=optimizer, sequential=False, max_retries=max_retries
            )
    # retry with random samples from the priors upon failure
    mll.train()
    original_state_dict = deepcopy(mll.model.state_dict())
    retry = 0
    while retry < max_retries:
        with warnings.catch_warnings(record=True) as ws:
            if retry > 0:  # use normal initial conditions on first try
                mll.model.load_state_dict(original_state_dict)
                sample_all_priors(mll.model)
            mll, _ = optimizer(mll, track_iterations=False, **kwargs)
            if not any(issubclass(w.category, OptimizationWarning) for w in ws):
                mll.eval()
                return mll
            retry += 1
            logging.log(logging.DEBUG, f"Fitting failed on try {retry}.")

    warnings.warn("Fitting failed on all retries.", OptimizationWarning)
    return mll.eval()
