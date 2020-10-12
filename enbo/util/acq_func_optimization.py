import torch
from torch import Tensor
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler, MCSampler
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition import AcquisitionFunction, qKnowledgeGradient
from botorch.optim import optimize_acqf
from gpytorch.models.gp import GP
import numpy as np
import os
from datetime import datetime
from .rollout import rollout_wrapper, UserData, rollout_wrapper_direct, CC, CG
from .global_variables import dtype, DIRECT_MAXT
from .glasses import estimate_lipschitz_constant
from .direct_optim import GlassesArgs, glasses_wrapper_direct
import DIRECT
from botorch.sampling.samplers import GaussHermiteSampler
from botorch.acquisition import qMultiStepLookahead
from .two_step_ei_envelope import TwoStepEIEnvelope
from typing import List
from botorch.acquisition.multi_step_lookahead import make_best_f
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.gen import gen_candidates_torch
from botorch.utils.sampling import draw_sobol_samples


def optimize_acq_func(acq_func: AcquisitionFunction, bounds=None, options=None):
    """Optimizes the acquisition function"""

    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=512,
        options=options,
    )
    new_x = candidates.detach()
    return new_x


def optimize_acq_func_and_get_observation(
    model: GP,
    method="EI",
    bounds=None,
    objective_func=None,
    remaining_budget=None,
    options=None,
):

    current_best = model.train_targets.max()

    if method == "EI":
        ei = ExpectedImprovement(model=model, best_f=current_best)
        # if objective_func is not None and objective_func.is_grid:
        #     unchosen_idx = objective_func.unchosen_idx
        #     with torch.no_grad():
        #         ei_values = ei(objective_func.x[unchosen_idx].unsqueeze(1))
        #     select_idx = torch.argmax(ei_values)
        #     return unchosen_idx[select_idx]
        # else:
        new_x = optimize_acq_func(
            ei, bounds=bounds, options={"seed": options.get("seed")}
        )
    elif "OKG" in method:
        _, num_fantasies = method.split(".")
        num_fantasies = int(num_fantasies)
        kg = qKnowledgeGradient(model=model, num_fantasies=num_fantasies)
        # if objective_func is not None and objective_func.is_grid:
        #     unchosen_idx = objective_func.unchosen_idx
        #     with torch.no_grad():
        #         ei_values = ei(objective_func.x[unchosen_idx].unsqueeze(1))
        #     select_idx = torch.argmax(ei_values)
        #     return unchosen_idx[select_idx]
        # else:
        new_x, _ = optimize_acqf(
            acq_function=kg,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={
                "seed": options.get("seed"),
                "log_nit_nfev": options.get("log_nit_nfev", None),
            },
        )

    elif ".rts." in method:  # Wu & Frazier (2019)
        params = method.split(".")  # e.g. 3.ts.n, where n is the number of fantasies
        q, _, num_y_samples = params[:3]
        q = int(q)
        num_y_samples = int(num_y_samples)
        method = "scipy" if len(params) < 4 else params[3]
        num_batches = 5 if len(params) < 5 else int(params[4])
        q, is_ending = _adapt_q(q, remaining_budget)
        if q == 1:
            return optimize_acq_func_and_get_observation(
                model, "EI", bounds, objective_func, remaining_budget, options
            )
        num_y_samples = int(num_y_samples)
        sampler = GaussHermiteSampler(
            num_samples=num_y_samples, resample=False, collapse_batch_dims=True
        )
        inner_sampler = SobolQMCNormalSampler(
            num_samples=options["num_inner_mc_samples"], seed=options["seed"]
        )
        two_step = TwoStepEIEnvelope(
            model=model,
            best_f=current_best,
            bounds=bounds,
            sampler=sampler,
            inner_sampler=inner_sampler,
            q1=q - 1,
            options={
                "seed": options["seed"],
                "method": method,
                "num_batches": num_batches,
            },
        )
        if method == "scipy":
            new_x, _ = optimize_acqf(
                acq_function=two_step,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=512,
                options={
                    "seed": options.get("seed"),
                    "maxiter": 150,
                    "log_nit_nfev": options.get("log_nit_nfev", None),
                },
            )
        else:
            # Xinit = gen_batch_initial_conditions(
            #     acq_function=two_step,
            #     bounds=bounds,
            #     q=1,
            #     num_restarts=2,
            #     raw_samples=10,
            #     options={"nonnegative": True},
            # )
            Xinit = draw_sobol_samples(
                bounds=bounds, n=50, q=1, seed=options.get("seed", None)
            )
            optimizer = torch.optim.SGD if method == "sgd" else torch.optim.Adam
            batch_candidates, batch_acq_values = gen_candidates_torch(
                initial_conditions=Xinit,
                acquisition_function=two_step,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                optimizer=optimizer,
                options={"maxiter": 500, "lr": 1.0, "scheduler_on": True, "gamma": 0.7},
                # options={"maxiter": 500},
                verbose=True,
            )
            best = torch.argmax(batch_acq_values.view(-1), dim=0)
            new_x = batch_candidates[best].detach()

    elif "ms" in method:  # [warm-start] [pseudo] multi-step
        # e.g. 3.wsms.n1.n2, where n1, n2 is the number of fantasies
        params = method.split(".")
        q = int(params[0])
        use_pseudo_ms = "pms" in params[1]
        ms = "pms" if use_pseudo_ms else "ms"
        method_opt = params[1].split(ms)
        use_warm_start = method_opt[0] == "ws"
        batch_limit = None
        if method_opt[1]:
            batch_limit = int(method_opt[1])

        q, is_ending = _adapt_q(q, remaining_budget)
        if q == 1:
            ei = ExpectedImprovement(model=model, best_f=current_best)
            new_x = optimize_acq_func(
                ei, bounds=bounds, options={"seed": options.get("seed")}
            )
            return new_x

        params, maxfun = _get_maxfun(params)
        num_fantasies, samplers = _construct_samplers(
            params[2:], q, use_pseudo_ms=use_pseudo_ms
        )

        from botorch.acquisition.multi_step_lookahead import warmstart_multistep

        batch_sizes = [q - 1] if use_pseudo_ms else [1] * (q - 1)
        valfunc_cls = [ExpectedImprovement] + (
            [qExpectedImprovement]
            if use_pseudo_ms and q > 2
            else [ExpectedImprovement] * (q - 1)
        )
        valfunc_argfacs = [make_best_f] * (2 if use_pseudo_ms else q)
        inner_mc_samples = (
            options["num_inner_mc_samples"] * (q - 1) if use_pseudo_ms else None
        )
        multi_step = qMultiStepLookahead(
            model=model,
            batch_sizes=batch_sizes,
            num_fantasies=None if samplers else num_fantasies,
            samplers=samplers,
            valfunc_cls=valfunc_cls,
            valfunc_argfacs=valfunc_argfacs,
            inner_mc_samples=[None, inner_mc_samples],
        )
        q_prime = multi_step.get_augmented_q_batch_size(1)
        num_restarts, raw_samples = 10, 512
        last_iter_optimizer = options.get("last_iter_optimizer", None)
        if not use_warm_start or last_iter_optimizer is None or is_ending:
            X_init_new = None
        else:
            X_init_new = warmstart_multistep(  # TODO: warmerstart_multistep
                acq_function=multi_step,
                bounds=bounds,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                full_optimizer=last_iter_optimizer,
            )
        tree, tree_vals = optimize_acqf(
            acq_function=multi_step,
            bounds=bounds,
            q=q_prime,
            num_restarts=num_restarts,
            raw_samples=raw_samples,  # , options=options
            batch_initial_conditions=X_init_new,
            return_best_only=False,
            return_full_tree=True,
            options={
                "log_nit_nfev": options.get("log_nit_nfev"),
                "seed": options.get("seed"),
                "maxfun": maxfun,
                "verbose": True,
                "batch_limit": batch_limit or num_restarts,
            },
        )
        options["last_iter_optimizer"] = tree
        best_tree_idx = torch.argmax(tree_vals)
        best_X_full = tree[best_tree_idx]
        new_x = multi_step.extract_candidates(best_X_full)

    elif ".EI" in method:  # qEI
        # define the qNEI acquisition module using a QMC sampler
        # e.g. 3.EI.sample
        params = method.split(".")
        q, _, pick_method = params[:3]
        q = int(q)
        q, is_ending = _adapt_q(q, remaining_budget)
        num_samples = options.get("num_mc_samples", 10000)
        maxfun = 15000
        if len(params) >= 4:
            if "_mf" in params[3]:
                num_samples, maxfun = params[3].split("_mf")
                maxfun = int(maxfun)
            else:
                num_samples = params[3]
            num_samples = int(num_samples) * q
        qmc_sampler = SobolQMCNormalSampler(
            num_samples=num_samples, seed=options["seed"]
        )
        qEI = qExpectedImprovement(
            model=model, best_f=current_best, sampler=qmc_sampler
        )
        ei = ExpectedImprovement(model=model, best_f=current_best)
        # optimize and get new observation
        new_x = optimize_batch_then_select_one(
            batch_acq_func=qEI,
            single_acq_func=ei,
            batch_size=q,
            bounds=bounds,
            pick_method=pick_method,
            options={
                "seed": options.get("seed"),
                "maxfun": maxfun,
                "log_nit_nfev": options.get("log_nit_nfev"),
            },
        )

    elif ".qEI" in method:  # qEI
        # define the qNEI acquisition module using a QMC sampler
        q = int(method.split(".")[0])
        if remaining_budget is not None:
            q = min(q, remaining_budget)
        qmc_sampler = SobolQMCNormalSampler(
            num_samples=options["num_mc_samples"], seed=options["seed"]
        )
        qEI = qExpectedImprovement(
            model=model, best_f=current_best, sampler=qmc_sampler
        )
        # optimize
        new_x, _ = optimize_acqf(
            acq_function=qEI,
            bounds=bounds,
            q=q,
            num_restarts=20,
            raw_samples=512,
            options={"seed": options.get("seed")},
        )

    elif "rollout" in method:
        method_args = method.split(".")  # e.g. 3.rollout.5 or 3.rollout.5.20
        q, num_y_samples = int(method_args[0]), int(method_args[2])
        if len(method_args) >= 4:
            direct_maxT = int(method_args[3])
        else:  # if not specified in method string, use global
            direct_maxT = DIRECT_MAXT
        q, is_ending = _adapt_q(q, remaining_budget)

        samples, weights = np.polynomial.hermite.hermgauss(num_y_samples)
        if objective_func is not None and objective_func.is_grid and "GG" in method:
            x_grid = objective_func.x
            unchosen_idx = objective_func.unchosen_idx
            rollout_utility = rollout_wrapper(
                None,
                model=model,
                best_f=current_best,
                bounds=bounds,
                x_grid=x_grid,
                indices=unchosen_idx,
                quadrature=(samples, weights),
                horizon=q,
                num_y_samples=num_y_samples,
            )
            select_idx = np.argmax(rollout_utility)
            return unchosen_idx[select_idx]
        else:
            # optimize with DIRECT
            mode = CG if objective_func is not None and objective_func.is_grid else CC
            x_grid = None if mode == CC else objective_func.x
            user_data = UserData(
                model=model,
                best_f=current_best,
                bounds=bounds,
                mode=mode,
                x_grid=x_grid,
                quadrature=(samples, weights),
                horizon=q,
                num_y_samples=num_y_samples,
            )
            now = datetime.now()
            timestamp = datetime.timestamp(now)
            direct_log_dir = "__tmp_direct__"
            if not os.path.exists(direct_log_dir):
                os.mkdir(direct_log_dir)
            if objective_func.is_grid:
                func_name = objective_func.dataname
            else:
                func_name = objective_func.__class__.__name__
            direct_log_file = (
                func_name
                + "_"
                + method
                + "_"
                + str(options["seed"])
                + "_"
                + str(timestamp)
            )
            direct_log_path = os.path.join(direct_log_dir, direct_log_file)
            new_x, fmin, ierror = DIRECT.solve(
                rollout_wrapper_direct,
                l=bounds[0].numpy(),
                u=bounds[1].numpy(),
                maxT=direct_maxT * objective_func.dim,
                logfilename=direct_log_path,
                user_data=user_data,
            )
            new_x = torch.tensor(new_x, dtype=dtype)
            os.remove(direct_log_path)

    elif "glasses" in method:

        method_args = method.split(".")  # e.g. 3.glasses or 3.glasses.20
        q = int(method_args[0])
        if len(method_args) >= 3 and method_args[2] != "0":
            direct_maxT = int(method_args[2])
        else:  # if not specified in method string, use global
            direct_maxT = DIRECT_MAXT
        q, is_ending = _adapt_q(q, remaining_budget)
        if len(method_args) >= 4 and method_args[3] == "initL":
            L = options["L"] or estimate_lipschitz_constant(model, bounds)
        else:
            L = estimate_lipschitz_constant(model, bounds)
        # optimize with DIRECT
        options = options or {}
        user_data = GlassesArgs(
            model=model,
            bounds=bounds,
            L=L,
            y0=current_best,
            horizon=q,
            num_mc_samples=options.get("num_mc_samples", 10000),
        )
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        direct_log_dir = "__tmp_direct__"
        if not os.path.exists(direct_log_dir):
            os.mkdir(direct_log_dir)
        if objective_func.is_grid:
            func_name = objective_func.dataname
        else:
            func_name = objective_func.__class__.__name__
        direct_log_file = (
            func_name + "_" + method + "_" + str(options["seed"]) + "_" + str(timestamp)
        )
        direct_log_path = os.path.join(direct_log_dir, direct_log_file)
        new_x, fmin, ierror = DIRECT.solve(
            glasses_wrapper_direct,
            l=bounds[0].numpy(),
            u=bounds[1].numpy(),
            maxT=direct_maxT * objective_func.dim,
            logfilename=direct_log_path,
            user_data=user_data,
        )
        new_x = torch.tensor(new_x, dtype=dtype)
        os.remove(direct_log_path)

    elif method == "random":
        if objective_func is not None and objective_func.is_grid:
            unchosen_idx = objective_func.unchosen_idx
            num = len(unchosen_idx)
            return unchosen_idx[torch.randperm(num)[0]]
        else:
            func_dim = bounds.shape[1]
            new_x = torch.rand(1, func_dim, device=bounds.device)
    else:
        print("method %s not supported" % method)

    return new_x


def optimize_batch_then_select_one(
    batch_acq_func=None,
    batch_size=2,
    single_acq_func: AcquisitionFunction = None,
    bounds=None,
    pick_method="best",
    options=None,
):
    """compute an optimal batch, then select the best point"""

    # optimize
    candidates, _ = optimize_acqf(
        acq_function=batch_acq_func,
        bounds=bounds,
        q=batch_size,
        #        num_restarts=min(40, max(20, 4 * batch_size + 12)),
        num_restarts=20,
        raw_samples=512,
        options=options,
    )

    # observe new values
    new_x = candidates.detach()
    acq_values = single_acq_func(new_x.unsqueeze(1))
    if len(acq_values) == 1:
        return new_x
    idx = 0
    if pick_method == "best":
        idx = torch.argmax(acq_values)
    elif pick_method == "sample":
        prob = acq_values.cpu().detach().numpy().squeeze()
        prob = prob / np.sum(prob)
        idx = np.random.choice(np.arange(batch_size), p=prob)
    return new_x[idx]


def _construct_samplers(sampler_params: List, q: int, use_pseudo_ms=False):
    # sampler_params: ["gh", n1, n2, ...] or [n1, n2, ...]
    num_fantasies = (
        [1] if use_pseudo_ms else [1] * (q - 1)
    )  # by default only use one sample for each stage
    index = 0 if sampler_params[0].isnumeric() else 1
    end_idx = index + 1 if use_pseudo_ms else index + q - 1
    for i, param in enumerate(sampler_params[index:end_idx]):
        num_fantasies[i] = int(param)
    samplers = None
    if sampler_params[0] == "gh":
        samplers = [
            GaussHermiteSampler(
                num_samples=nf, resample=False, collapse_batch_dims=True
            )
            for nf in num_fantasies
        ]
    elif sampler_params[0] == "sobol_collapse_false":
        samplers: List[MCSampler] = [
            SobolQMCNormalSampler(
                num_samples=nf, resample=False, collapse_batch_dims=False
            )
            for nf in num_fantasies
        ]
    return num_fantasies, samplers


def _adapt_q(q, remaining_budget):

    is_ending = False
    if remaining_budget is None:
        return q, is_ending

    if q == 0:
        q = remaining_budget
        is_ending = True
    else:
        is_ending = remaining_budget < q
        q = min(q, remaining_budget)
    return q, is_ending


def _get_maxfun(params):
    # e.g. 3.wstbps8.gh.10.5_mf200
    # then params[-1] = "5_mf200"
    maxfun = 15000  # default in _minimize_lbfgsb
    if "_mf" in params[-1]:
        opt_str = params[-1].split("_mf")
        params[-1] = opt_str[0]
        maxfun = int(opt_str[1])
    return params, maxfun
