#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
BoTorch settings.
"""

import typing  # noqa F401
import warnings

from .exceptions import BotorchWarning


class _Flag:
    r"""Base class for context managers for a binary setting."""

    _state: bool = False

    @classmethod
    def on(cls) -> bool:
        return cls._state

    @classmethod
    def off(cls) -> bool:
        return not cls._state

    @classmethod
    def _set_state(cls, state: bool) -> None:
        cls._state = state

    def __init__(self, state: bool = True) -> None:
        self.prev = self.__class__.on()
        self.state = state

    def __enter__(self) -> None:
        self.__class__._set_state(self.state)

    def __exit__(self, *args) -> None:
        self.__class__._set_state(self.prev)


class propagate_grads(_Flag):
    r"""Flag for propagating gradients to model training inputs / training data.

    When set to `True`, gradients will be propagated to the training inputs.
    This is useful in particular for propating gradients through fantasy models.
    """

    _state: bool = False


def suppress_botorch_warnings(suppress: bool) -> None:
    r"""Set botorch warning filter.

    Args:
        state: A boolean indicating whether warnings should be prints
    """
    warnings.simplefilter("ignore" if suppress else "default", BotorchWarning)


class debug(_Flag):
    r"""Flag for printing verbose BotorchWarnings.

    When set to `True`, verbose `BotorchWarning`s will be printed for debuggability.
    Warnings that are not subclasses of `BotorchWarning` will not be affected by
    this context_manager.
    """

    _state: bool = False
    suppress_botorch_warnings(suppress=not _state)

    @classmethod
    def _set_state(cls, state: bool) -> None:
        cls._state = state
        suppress_botorch_warnings(suppress=not cls._state)


class validate_input_scaling(_Flag):
    r"""Flag for validating input normalization/standardization.

    When set to `True`, standard botorch models will validate (up to reasonable
    tolerance) that
    (i) none of the inputs contain NaN values
    (ii) the training data (`train_X`) is normalized to the unit cube
    (iii) the training targets (`train_Y`) are standardized (zero mean, unit var)
    No checks (other than the NaN check) are performed for observed variances
    (`train_Y_var`) at this point.
    """

    _state: bool = True
