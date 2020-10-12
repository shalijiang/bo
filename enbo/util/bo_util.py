import time
import numpy as np
import torch
import gpytorch
import torch.tensor as Tensor

# from botorch.models import SingleTaskGP, FixedNoiseGP
from .gp_regression import SingleTaskGP, FixedNoiseGP

from botorch import fit_gpytorch_model
from botorch.optim.utils import _get_extra_mll_args
from copy import deepcopy
import datetime

from .utils import DataTransformer
from .utils import standardize

from .global_variables import device, dtype
from .acq_func_optimization import optimize_acq_func_and_get_observation
from .glasses import estimate_lipschitz_constant
from .gp_samples import GPSqeModel

NOISE_SE = 0.01
train_yvar = Tensor(NOISE_SE ** 2, device=device)


def initialize_data(objective_func, num_initial=10):
    # generate training data
    func_dim = objective_func.dim
    bounds = objective_func.bounds.to(device)
    transform = DataTransformer(bounds)
    if objective_func.is_grid:
        train_idx = torch.randperm(objective_func.n, device=device)[:num_initial]
        raw_train_x, raw_train_y = objective_func.get_data(train_idx)
        train_x = transform.from_original_to_normalized(raw_train_x)
        train_y = standardize(raw_train_y)
    else:
        train_x = torch.rand(num_initial, func_dim, device=device, dtype=dtype)
        raw_train_x = transform.from_normalized_to_original(train_x)
        raw_train_y = objective_func.evaluate(raw_train_x)
        train_y = standardize(raw_train_y)
        train_idx = None
    if hasattr(objective_func, "funcname") and objective_func.funcname.startswith(
        "gp_sample"
    ):
        train_y = raw_train_y
    print("\ninitial observed values: ")
    print(raw_train_y)
    best_observed_value = raw_train_y.max().item()  # maximization problem
    print("initial best value: ", best_observed_value)
    print("optimal value: ", objective_func.optimal_value)
    # define models for objective and constraint

    return (
        train_x,
        train_y.unsqueeze(-1),
        best_observed_value,
        transform,
        raw_train_x,
        raw_train_y.unsqueeze(-1),
        train_idx,
    )


def initialize_model(train_x, train_y, fix_noise=False, state_dict=None, sim_exp=False):
    # define models for objective and constraint
    if sim_exp:
        model = GPSqeModel(train_x, train_y)
        if state_dict is None:
            print("initial model parameters")
            for name, param in model.named_parameters():
                print(name, param)
    else:
        if fix_noise:
            model = FixedNoiseGP(train_x, train_y, train_yvar.expand_as(train_y)).to(
                train_x
            )
        else:
            model = SingleTaskGP(train_x, train_y)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None and not sim_exp:
        model.load_state_dict(state_dict)

    return mll, model


def multi_start_fit_gpytorch_model(
    mll, num_restart=10, max_num_restart=20, method="L-BFGS-B", options={"disp": False}
):
    """
    multi-restart fit_gpytorch_model.

    :param mll:
    :param num_restart: minimum number of restarts
    :param max_num_restart: if all num_restart failed, increase until max_num_restart
    :param method: optimize method
    :param options:
    """
    max_marginal_likelihood = torch.tensor(
        float("-inf"), device=mll.model.train_targets.device
    )
    state = deepcopy(mll.state_dict())
    i = 0
    succeed_at_least_once = False
    while i < num_restart:
        mll.model.covar_module.base_kernel.sample_from_prior("lengthscale_prior")
        mll.model.covar_module.sample_from_prior("outputscale_prior")
        if "noise_prior" in mll.model.likelihood.noise_covar._priors:
            mll.model.likelihood.noise_covar.sample_from_prior("noise_prior")
        i += 1
        if (
            i == num_restart
            and not succeed_at_least_once
            and num_restart < max_num_restart
        ):
            num_restart += 5
        try:
            fit_gpytorch_model(mll, method=method, options=options)
            succeed_at_least_once = True
        except Exception as e:
            err_msg = "restart %d: " % i + str(e)
            print(err_msg)
            continue
        train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
        mll.train()
        output = mll.model(*train_inputs)
        args = [output, train_targets] + _get_extra_mll_args(mll)
        marginal_likelihood = mll(*args).sum()
        if marginal_likelihood > max_marginal_likelihood:
            max_marginal_likelihood = marginal_likelihood
            state = deepcopy(mll.state_dict())
            print("restart %d has higher mll: %f" % (i, marginal_likelihood.item()))
            # pprint(list(mll.model.named_parameters()))

    mll.load_state_dict(state)


def bo_loop(
    objective_func,
    method,  # which acquisition function
    num_bo_iters,  # number of BO iterations
    num_initial=5,  # number of initial obervations
    seed=None,  # random seed for generating the initial obervations
    num_restart=3,  # for multi_start_git_gpytorch_model
    max_num_restart=10,  # for multi_start_git_gpytorch_model
    max_iter_for_opt_acq=500,  # max number of iteration for optimizing acquisition function
    do_simple_init=True,  # how to initialize the optimization of the acquisition function
    opt_method="L-BFGS-B",  # method for optimizing the acquisition function
    num_mc_samples=500,
    num_inner_mc_samples=500,
    fix_noise=False,
    verbose=True,
    log_file=None,
    opt_log_file=None,
    debug_mode=False,
):
    if objective_func.is_grid and "qEI" in method:
        raise NotImplementedError("batch_size > 1 not supported for discrete functions")
    import logging

    if log_file is not None:
        logging.basicConfig(filename=log_file)
    best_observed = []
    # call helper function to initialize model
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    train_x, train_y, best_value, transformer, raw_train_x, raw_train_y, train_idx = initialize_data(
        objective_func, num_initial=num_initial
    )

    func_name = (
        objective_func.dataname
        if objective_func.is_grid
        else objective_func.__class__.__name__
    )
    sim_exp = func_name.startswith("gp_sample")
    mll, model = initialize_model(
        train_x, train_y, fix_noise=fix_noise, sim_exp=sim_exp
    )

    header_msg = (
        str(datetime.datetime.now())
        + f"\t{func_name} {objective_func.dim} {method} trial {seed} starting..."
    )
    logging.warning(header_msg)

    info = (
        f" {func_name} {method} trial {seed} iteration {0:>2}: "
        f"observed value = {best_value:>10f}   "
        f"best_value = {best_value:>10f}, "
        f"time = {0:>5f}."
    )
    logging.critical(info)
    if verbose:
        print(info)

    best_observed.append(best_value)

    bounds = torch.tensor(
        [[0.0, 1.0] for _ in range(objective_func.dim)], device=device, dtype=dtype
    ).t()
    # optimize_on_grid = objective_func.is_grid and (
    #     method in ["random", "EI"] or "GG" in method
    # )
    optimize_on_grid = False
    time_used = np.zeros(num_bo_iters)
    # run num_bo_iters rounds of BayesOpt after the initial random batch
    initL = None
    options = {
        "simple_init": do_simple_init,
        "maxiter": max_iter_for_opt_acq,
        "num_mc_samples": num_mc_samples,
        "num_inner_mc_samples": num_inner_mc_samples,
        "log_nit_nfev": opt_log_file,
    }

    batch_size = int(method.split(".")[0]) if "qEI" in method else 1
    for iteration in range(1, num_bo_iters + 1, batch_size):

        t0 = time.time()
        options["seed"] = (
            seed * num_bo_iters + iteration
        )  # add seed to ensure sobol sequence is reproduciable

        remaining_budget = num_bo_iters - iteration + 1  # including this iteration
        try:
            # fit the model
            if method != "random" and not sim_exp:
                multi_start_fit_gpytorch_model(
                    mll,
                    num_restart=num_restart,
                    max_num_restart=max_num_restart,
                    method=opt_method,
                )
                if iteration == 1 and "glasses" in method:
                    initL = estimate_lipschitz_constant(model, bounds)
                options["L"] = initL

            # pprint(list(mll.named_parameters()))
            if optimize_on_grid:
                objective_func.set_chosen_idx(train_idx)
                new_x_idx = optimize_acq_func_and_get_observation(
                    model,
                    method,
                    bounds=bounds,
                    objective_func=objective_func,
                    remaining_budget=remaining_budget,
                    options=options,
                )
            else:
                new_x = optimize_acq_func_and_get_observation(
                    model,
                    method,
                    bounds=bounds,
                    objective_func=objective_func,
                    remaining_budget=remaining_budget,
                    options=options,
                )
        except Exception as e:
            if debug_mode:
                raise (e)
            error_msg = (
                "\nwarning: iteration %d resulted in an error, switched to random sampling\n"
                % iteration
            )
            error_msg = error_msg + str(e)
            print(error_msg)

            # theta = mll.state_dict()
            # raw_length_scales = Tensor([-2.] * objective_func.dim)
            # theta['model.covar_module.base_kernel.raw_lengthscale'].copy_(raw_length_scales)
            if optimize_on_grid:
                objective_func.set_chosen_idx(train_idx)
                new_x_idx = optimize_acq_func_and_get_observation(
                    model,
                    method="random",
                    bounds=bounds,
                    objective_func=objective_func,
                    options=options,
                )
            else:
                new_x = optimize_acq_func_and_get_observation(
                    model, method="random", bounds=bounds, options=options
                )

        # get observation
        if optimize_on_grid:
            new_raw_x, new_raw_y = objective_func.get_data(new_x_idx)
            new_x = transformer.from_original_to_normalized(new_raw_x)
            train_idx = torch.cat((train_idx, new_x_idx.view(1)))
        else:
            new_raw_x = transformer.from_normalized_to_original(new_x)
            if objective_func.is_grid and objective_func.interpolater is None:
                objective_func.set_chosen_idx(train_idx)
                new_x_idx = objective_func.find_closest(new_raw_x)
                new_raw_x, new_raw_y = objective_func.get_data(new_x_idx)
                train_idx = torch.cat((train_idx, new_x_idx.view(1)))
            else:
                new_raw_y = objective_func.evaluate(new_raw_x)
        options["obversed_y"] = new_raw_y
        options["stardardize_y"] = (raw_train_y.mean(), raw_train_y.std())
        # update training points
        raw_train_x = torch.cat((raw_train_x, new_raw_x.view(-1, objective_func.dim)))
        raw_train_y = torch.cat((raw_train_y, new_raw_y.view([-1, 1])))

        train_x = torch.cat((train_x, new_x.view(-1, objective_func.dim)))
        if sim_exp:
            train_y = raw_train_y
            options["stardardize_y"] = None
        else:
            train_y = standardize(raw_train_y)

        # update progress
        best_value = max(best_value, torch.max(new_raw_y).item())
        best_observed.append(best_value)

        # reinitialize the model so it is ready for fitting on next iteration
        mll, model = initialize_model(
            train_x,
            train_y,
            fix_noise=fix_noise,
            state_dict=model.state_dict(),
            sim_exp=sim_exp,
        )

        t1 = time.time()
        time_used[iteration - 1] = t1 - t0
        info = (
            f" {func_name} {method} trial {seed} iteration {iteration:>2}: "
            f"observed value = {new_raw_y.numpy()}   "
            f"best_value = {best_value:>10f}, "
            f"time = {t1 - t0:>5f}."
        )
        logging.critical(info)
        if verbose:
            print(info)
        else:
            print(".")

    return best_observed, model, raw_train_x, raw_train_y, time_used
