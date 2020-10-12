import torch
import numpy as np


class DataTransformer(object):
    def __init__(self, bounds):
        self.range = (bounds[1] - bounds[0]).view(1, -1)
        self.lower_bound = bounds[0].view(1, -1)
        self.bounds = bounds

    def from_normalized_to_original(self, x_normalized):
        """
        transform from [0,1] to [lower, upper]
        :param x_normalized:
        :return:
        """
        return x_normalized * self.range + self.lower_bound

    def from_original_to_normalized(self, x_original):
        """
        transform from [lower, upper] to [0,1]
        :param x_original:
        :return:
        """
        return (x_original - self.lower_bound) / self.range


def standardize(y: torch.Tensor):
    return (y - y.mean()) / y.std()


def compute_gap(best_observed, optimal_value):
    # best_observed is a list
    # optimal_value is the maximum function value
    y0 = best_observed[0]  # initial best observation
    gap = (np.array(best_observed[1:]) - y0) / (optimal_value - y0)
    return gap


def trim_legend(method):
    method = method.replace("sample", "s")
    method = method.replace("best", "b")
    method = method.replace("rollout", "R")
    method = method.replace("glasses.20", "G")
    method = method.replace("glasses.0", "G")
    method = method.replace(".initL", "")
    method = method.replace("ts.10.1", "PMS.10")
    # method = method.replace(".gh.10.5.3", "")
    # method = method.replace(".gh.10.5", "")
    # method = method.replace(".gh.10", "")
    method = method.replace("rts", "ETS")
    method = method.replace("ts", "PMS")
    # method = method.replace("2.wsms.gh.10", '2-step-EI')
    # method = method.replace("3.wsms.gh.10.5", '3-step-EI')
    # method = method.replace("4.wsms.gh.10.5.3", '4-step-EI')
    return method
