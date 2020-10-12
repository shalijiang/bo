#!/usr/bin/env python
# coding: utf-8
import os

# to ensure each job only uses one CPU
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pickle
import warnings
import sys
import datetime
from util.bo_util import bo_loop
from util import hyper_tuning_functions_on_grid as grid_func
from util.test_functions import benchmark_funcs
from util.gp_samples import GPSamples
from util.global_variables import debug_mode
import shutil
from scipy.interpolate import LinearNDInterpolator
from sklearn.ensemble import RandomForestRegressor
# warnings.filterwarnings('ignore')
verbose = 1
debug_mode = 0
print_opt_log = True
if debug_mode:
    ##### for debugging  ######
    objective_func_name = 'svm_on_grid.interpolater' # 'branin' # 'dropwave'  # "branin"
    # objective_func_name = 'svm_on_grid'
    # objective_func_name = "branin"
    method =  '3.wsms.gh.3.5' #'2.rollout.2' #'2.EI.best'  # EI
    method = "2.wsms.gh.10"
    # method =  '3.ms.gh.10.5' #'2.rollout.2' #'2.EI.best'  # EI
    # method = "3.ts.10"
    # # method = "3.EI.sample"
    method = "EI"
    # # method = "3.tss.10"
    # method = "2.tbps.gh.10"
    # method = "3.tbps.gh.10.5"
    # method = "4.tbps.gh.10.5.3"
    num_iter_scale = 2  # 20
    num_init_scale = 2  # 3
    seed = '1'  # 1
    save_dir = './debug_results'
    log_dir = './debug_log'
    try:
        shutil.rmtree(save_dir)
    except:
        pass
else:
    #example: python run_bo_task.py branin EI 10 3 1
    objective_func_name = str(sys.argv[1])  # "branin"
    method = str(sys.argv[2])  # EI
    num_iter_scale = int(sys.argv[3])  # 20
    num_init_scale = int(sys.argv[4])  # 3
    seed: str = sys.argv[5]  # 1 or 1_30
    save_dir = './results' if len(sys.argv) < 7 else sys.argv[6]
    log_dir = './log' if len(sys.argv) < 8 else sys.argv[7]
    os.environ['USE_CUDA'] = 'NO' if len(sys.argv) < 9 else sys.argv[8]

from util.global_variables import (num_mc_samples,
                                   num_inner_mc_samples,
                                   max_iter_for_opt_acq,
                                   opt_method,
                                   MAXIMIZE,
                                   device)

if 'on_grid' in objective_func_name:
    interpolater = None
    if "interpolate" in objective_func_name:
        objective_func_name = objective_func_name.split(".")[0]
        # interpolater = LinearNDInterpolator
        interpolater = RandomForestRegressor()
    objective_func = grid_func.HyperTuningGrid(objective_func_name,
                                               maximize=MAXIMIZE,
                                               interpolater=interpolater,
                                               device=device)
elif "gp_sample" in objective_func_name:
    gp_sample_seed = int(objective_func_name[9:])
    objective_func = GPSamples(seed=gp_sample_seed, maximize=MAXIMIZE)
    print("model generating the data:")
    for name, param in objective_func.model.named_parameters():
        print(name, param)
else:
    objective_func = benchmark_funcs[objective_func_name]

fix_noise = False if objective_func.is_grid and interpolater is None else True
optimal_value = objective_func.optimal_value

# average over multiple trials
func_dim = objective_func.dim
num_bo_iters = num_iter_scale * func_dim
num_initial = func_dim * num_init_scale
if "gp_sample" in objective_func_name:
    num_bo_iters = 15
    num_initial = 1
    if debug_mode:
        num_bo_iters = 2

# id_str = objective_func_name + '_%d_samples_%d_iters_init_scale_%d' % (
#     num_mc_samples, max_iter_for_opt_acq, num_init_scale)
id_str = f'{objective_func_name}_{method}_num_iter_{num_bo_iters}_num_init_{num_initial}_seed_{seed}'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    print('Directory', save_dir, 'created')
save_path = os.path.join(save_dir, id_str)
if os.path.exists(save_path):
    print("results already exists!")
    exit(1)

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    print('Directory', log_dir, 'created')
log_path = os.path.join(log_dir, id_str)
opt_log_path = os.path.join(log_dir, "opt_" + id_str) if print_opt_log else None

currentDT = datetime.datetime.now()
print(str(currentDT))
print(id_str)

if seed.isnumeric():
    seeds = [int(seed)]
else:
    start, end = seed.split('_')
    seeds = list(range(int(start), int(end)+1))
best_observed_all = []
raw_x_all = []
raw_y_all = []
chosen_locations = []
observed_func_values = []
times = []
for repeat in seeds:
    print(str(datetime.datetime.now()), 'seed', repeat)
    results = bo_loop(objective_func,
                      method=method,
                      num_bo_iters=num_bo_iters,
                      num_initial=num_initial,
                      opt_method=opt_method,
                      num_mc_samples=num_mc_samples,
                      num_inner_mc_samples=num_inner_mc_samples,
                      seed=repeat,
                      verbose=verbose,
                      fix_noise=fix_noise,
                      log_file=log_path,
                      opt_log_file=opt_log_path,
                      debug_mode=debug_mode,
                      )
    best_observed, model, raw_x, raw_y, time = results
    best_observed_all.append(best_observed)
    chosen_locations.append(model.train_inputs[0].cpu().numpy())
    observed_func_values.append(model.train_targets.cpu().numpy())
    raw_x_all.append(raw_x)
    raw_y_all.append(raw_y)
    times.append(time)

print('\nsaving results to %s' % save_path)
with open(save_path, 'wb') as f:
    pickle.dump({"x": chosen_locations,
                 "y": observed_func_values,
                 "raw_x": raw_x_all,
                 "raw_y": raw_y_all,
                 "accmax_y": best_observed_all,
                 "time": times},
                f)
currentDT = datetime.datetime.now()
print(str(currentDT) + " " + id_str)
print('done')

