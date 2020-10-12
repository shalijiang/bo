import torch
from torch import Tensor
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models import SingleTaskGP
from rollout import rollout, rollout_quad
import warnings
import time
import pickle

warnings.filterwarnings("ignore")

bound = 0.5
bounds = Tensor([[-bound, bound], [-bound, bound]]).t()
n1 = 5
idx1 = Tensor(range(n1)).type(torch.long)
idx_tuples = torch.cartesian_prod(idx1, idx1)
x1 = torch.linspace(-bound, bound, n1)
x = torch.cartesian_prod(x1, x1)
n = x.shape[0]
num_init = 4

train_idx = list(np.random.permutation(n)[:num_init])
train_idx_tuples = idx_tuples[train_idx]
train_x = x[train_idx]

model = SingleTaskGP(train_x, Tensor([0] * num_init))

model.covar_module.base_kernel.lengthscale = 0.4
model.covar_module.outputscale = 1.0
model.likelihood.noise = 0.0001
model.eval()
with gpytorch.settings.debug(False):
    model.train()
    y_prior = model(x)
    model.eval()
torch.manual_seed(0)
y = y_prior.sample()

train_y = y[train_idx]
y_best = torch.max(train_y).item()
print(train_x)
print(train_y)
model.set_train_data(train_x, train_y, strict=False)
model.eval()


f, ax = plt.subplots(2, 2, figsize=(12, 12))

## compute EI
expected_improvement = ExpectedImprovement(model, best_f=y_best)

with torch.no_grad():
    y_post = model(x)
    ax[0, 0].contour(x1, x1, y.view(n1, n1))
    ax[0, 0].set_title("prior")
    ax[0, 1].contour(x1, x1, y_post.mean.view(n1, n1))
    ax[0, 1].plot(
        x1[train_idx_tuples[:, 0]].numpy(), x1[train_idx_tuples[:, 1]].numpy(), "rs"
    )
    ax[0, 1].set_title("posterior")
    ei_values = expected_improvement(x.unsqueeze(1))
    ax[1, 0].contour(x1, x1, ei_values.view(n1, n1))
    ax[1, 0].set_title("EI")
## compute two-step EI
two_step_ei = np.zeros((3, n))
times = np.zeros((3, n))
num_y_samples = 5
samples, weights = np.polynomial.hermite.hermgauss(num_y_samples)
for i in range(n):
    print("point", i)
    this_x = x[i]

    # start = time.time()
    # two_step_ei[0, i] = rollout(this_x, model,
    #                             best_f=y_best,
    #                             bounds=Tensor([-bound, bound]).view(-1, 1),
    #                             horizon=2,
    #                             quadrature='qmc',
    #                             num_y_samples=num_y_samples,
    #                             )
    # end = time.time()
    # times[0, i] = end - start
    # print('qmc', end - start)

    start = time.time()
    two_step_ei[1, i] = rollout(
        this_x,
        model,
        best_f=y_best,
        bounds=bounds,
        horizon=4,
        # x_grid=x,
        # idx=i,
        quadrature=(samples, weights),
        num_y_samples=num_y_samples,
    )
    end = time.time()
    times[1, i] = end - start
    print("gauss-hermite", end - start)

    # start = time.time()
    # two_step_ei[2, i] = rollout_quad(this_x, model,
    #                                  best_f=y_best,
    #                                  bounds=Tensor([-bound, bound]).view(-1, 1),
    #                                  horizon=2,
    #                                  )
    # end = time.time()
    # times[2, i] = end - start
    # print('adap-gauss', end - start)

mean_time = times.mean(axis=1)
with torch.no_grad():
    # ax[1].plot(x.squeeze().numpy(), two_step_ei[0], label='two-step EI qmc %.2fs' % mean_time[0])
    ax[1, 1].contour(x1.numpy(), x1.numpy(), two_step_ei[1].reshape(n1, n1))
    ax[1, 1].set_title("two-step-ei")
    # ax[1].plot(x.squeeze().numpy(), two_step_ei[2], label='two-step EI adap-gauss %.2fs' % mean_time[2])


print(times.mean(axis=1))
with open("rollout2d_test_results", "wb") as f:
    pickle.dump({"time": times, "ei": two_step_ei}, f)
plt.show()
