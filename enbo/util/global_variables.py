import os
import torch

MAXIMIZE = True
if "USE_CUDA" in os.environ and os.environ["USE_CUDA"] == "YES":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

dtype = torch.float

# number of monte carlo samples for computing the acquisition function
num_mc_samples = 10000
num_inner_mc_samples = 512

max_iter_for_opt_acq = 500
opt_method = "L-BFGS-B"
DIRECT_MAXT = 20
debug_mode = 0 #False
