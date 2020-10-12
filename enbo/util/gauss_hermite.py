#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Sampler modules to be used with MC-evaluated acquisition functions.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import numpy as np

from torch import Tensor
from torch.nn import Module
from torch.quasirandom import SobolEngine

from botorch.exceptions import UnsupportedError
from botorch.posteriors import Posterior
from botorch.utils.sampling import manual_seed
from botorch.acquisition.monte_carlo import MCSampler


class GaussHermiteSampler(MCSampler):
    r"""Sampler for Gaussian hermite base samples.

    Example:
        >>> sampler = GaussHermiteSampler(1000, seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def __init__(
        self,
        num_samples: int,
        resample: bool = False,
        seed: Optional[int] = None,
        collapse_batch_dims: bool = True,
    ) -> None:
        r"""Sampler for quasi-MC base samples using Sobol sequences.

        Args:
            num_samples: The number of samples to use.
            resample: If `True`, re-draw samples in each `forward` evaluation -
                this results in stochastic acquisition functions (and thus should
                not be used with deterministic optimization algorithms).
            seed: The seed for the RNG. If omitted, use a random seed.
            collapse_batch_dims: If True, collapse the t-batch dimensions to
                size 1. This is useful for preventing sampling variance across
                t-batches.
        """
        super().__init__()
        self._sample_shape = torch.Size([num_samples])
        self.collapse_batch_dims = collapse_batch_dims
        self.resample = resample
        self.seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()

    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        r"""Generate quasi-random Normal base samples (if necessary).

        This function will generate a new set of base samples and set the
        `base_samples` buffer if one of the following is true:

        - `resample=True`
        - the MCSampler has no `base_samples` attribute.
        - `shape` is different than `self.base_samples.shape` (if
          `collapse_batch_dims=True`, then batch dimensions of will be
          automatically broadcasted as necessary)

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        if (
            self.resample
            or not hasattr(self, "base_samples")
            or self.base_samples.shape[-2:] != shape[-2:]
            or (not self.collapse_batch_dims and shape != self.base_samples.shape)
        ):
            output_dim = shape[-2:].numel()
            assert (
                output_dim == 1
            ), f"only output_dim = 1 is supported, but got {output_dim}"
            n = shape[:-2].numel()
            base_samples, base_weights = np.polynomial.hermite.hermgauss(n)
            base_samples = (
                torch.tensor(
                    base_samples, device=posterior.device, dtype=posterior.dtype
                ).view(n, output_dim)
                * 1.4142135623730951  # sqrt(2.0)
            )
            base_weights = (
                torch.tensor(
                    base_weights, device=posterior.device, dtype=posterior.dtype
                )
                / 1.7724538509055  # sqrt(pi)
            ).view(-1, 1)
            self.seed += 1
            base_samples = base_samples.view(shape)
            self.register_buffer("base_samples", base_samples)
            self.register_buffer("base_weights", base_weights)

        elif self.collapse_batch_dims and shape != posterior.event_shape:
            self.base_samples = self.base_samples.view(shape)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover
        if self.base_samples.dtype != posterior.dtype:
            self.to(dtype=posterior.dtype)
