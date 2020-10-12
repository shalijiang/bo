from .glasses import glasses
import torch


class GlassesArgs(object):
    def __init__(self, model, bounds, L, y0, horizon, num_mc_samples, sample_seed=None):
        self.model = model
        self.bounds = bounds
        self.L = L
        self.y0 = y0
        self.horizon = horizon
        self.num_mc_samples = num_mc_samples
        self.sample_seed = sample_seed


def glasses_wrapper_direct(x, user_data: GlassesArgs):
    value = glasses(
        torch.Tensor(x),
        model=user_data.model,
        bounds=user_data.bounds,
        L=user_data.L,
        y0=user_data.y0,
        horizon=user_data.horizon,
        num_mc_samples=user_data.num_mc_samples,
        sample_seed=user_data.sample_seed,
    )
    return -value, 0  # DIRECT.solve minimization problem
