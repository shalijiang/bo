#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.gen import gen_candidates_scipy, get_best_candidates
from gpytorch import settings as gpt_settings

from ..test_gen import TestBaseCandidateGeneration


EPS = 1e-8


class TestRandomRestartOptimization(TestBaseCandidateGeneration):
    def test_random_restart_optimization(self):
        for double in (True, False):
            self._setUp(double=double)
            with gpt_settings.debug(False):
                best_f = self.model(self.train_x).mean.max().item()
            qEI = qExpectedImprovement(self.model, best_f=best_f)
            bounds = torch.tensor([[0.0], [1.0]]).type_as(self.train_x)
            batch_ics = torch.rand(2, 1).type_as(self.train_x)
            batch_candidates, batch_acq_values = gen_candidates_scipy(
                initial_conditions=batch_ics,
                acquisition_function=qEI,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                options={"maxiter": 3},
            )
            candidates = get_best_candidates(
                batch_candidates=batch_candidates, batch_values=batch_acq_values
            )
            self.assertTrue(-EPS <= candidates <= 1 + EPS)
