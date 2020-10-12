#!/usr/bin/env python
# coding: utf-8
import os

# to ensure each job only uses one CPU
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import warnings
import datetime
from util.bo_util import bo_loop
from util import hyper_tuning_functions_on_grid as grid_func
from util.test_functions import benchmark_funcs
from util.gp_samples import GPSamples
from util.global_variables import debug_mode
from sklearn.ensemble import RandomForestRegressor

# warnings.filterwarnings('ignore')
from util.global_variables import (
    num_mc_samples,
    num_inner_mc_samples,
    max_iter_for_opt_acq,
    opt_method,
    MAXIMIZE,
    device,
)


def run_bo_task_func(
    objective_func_name: str,  # e.g., "branin", "svm_on_grid", "svm_on_grid.interpolater"
    method: str,  # BO policy, e.g., "EI", "2.wsms.gh.10"
    num_iter_scale: int = 20,  # num_iter_scale * d would be the number of BO iterations
    num_init_scale: int = 2,  # num_init_scale * d would be the number of initial random observations
    seed: int = 1,  # random seed
    save_dir="./results",  # pickled results of BO saved here
    log_dir="./log",  # log file directory
    verbose=True,
    print_opt_log=True,  # print #iter and #eval in optimize_acqf
        # (see this diff: https://github.com/shalijiang/botorch_dev/commit/f9d0030463b569fa0592061357bc488d16cd1644)
):

    if "on_grid" in objective_func_name:
        interpolater = None
        if "interpolate" in objective_func_name:
            objective_func_name = objective_func_name.split(".")[0]
            interpolater = RandomForestRegressor()
        objective_func = grid_func.HyperTuningGrid(
            objective_func_name,
            maximize=MAXIMIZE,
            interpolater=interpolater,
            device=device,
        )
    elif "gp_sample" in objective_func_name:  # functions sampled from a GP
        gp_sample_seed = int(objective_func_name[9:])
        objective_func = GPSamples(seed=gp_sample_seed, maximize=MAXIMIZE)
        print("model generating the data:")
        for name, param in objective_func.model.named_parameters():
            print(name, param)
    else:  # synthetic functions
        objective_func = benchmark_funcs[objective_func_name]

    fix_noise = False if objective_func.is_grid and interpolater is None else True

    func_dim = objective_func.dim
    num_bo_iters = num_iter_scale * func_dim
    num_initial = func_dim * num_init_scale
    if "gp_sample" in objective_func_name:  # follow Lam et al. 2016 setting
        num_bo_iters = 15
        num_initial = 1

    id_str = f"{objective_func_name}_{method}_num_iter_{num_bo_iters}_num_init_{num_initial}_seed_{seed}"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("Directory", save_dir, "created")

    save_path = os.path.join(save_dir, id_str)
    if os.path.exists(save_path):
        print("results already exists!")
        return 1

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        print("Directory", log_dir, "created")

    log_path = os.path.join(log_dir, id_str)
    print("log saved in" + log_path)
    # opt_log_path save the #iter and #eval in optimize_acq
    opt_log_path = os.path.join(log_dir, "opt_" + id_str) if print_opt_log else None

    current_datetime = datetime.datetime.now()
    print("start", str(current_datetime), id_str)
    results = bo_loop(
        objective_func,
        method=method,
        num_bo_iters=num_bo_iters,
        num_initial=num_initial,
        opt_method=opt_method,
        num_mc_samples=num_mc_samples,
        num_inner_mc_samples=num_inner_mc_samples,
        seed=seed,
        verbose=verbose,
        fix_noise=fix_noise,
        log_file=log_path,
        opt_log_file=opt_log_path,
        debug_mode=debug_mode,
    )
    best_observed, model, raw_x, raw_y, time = results

    print("\nsaving results to %s" % save_path)
    with open(save_path, "wb") as f:
        pickle.dump(
            {"raw_x": raw_x, "raw_y": raw_y, "accmax_y": best_observed, "time": time}, f
        )
    current_datetime = datetime.datetime.now()
    print("finish", str(current_datetime) + " " + id_str)
    print("done")
    return results
