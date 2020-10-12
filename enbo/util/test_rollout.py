import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models import SingleTaskGP
from rollout import rollout, rollout_quad
import warnings
import time
import pickle

warnings.filterwarnings("ignore")

bound = 1.0
n = 50
x = torch.linspace(-bound, bound, n).view(-1, 1)
train_idx = [np.round(n / 3), np.round(n * 2 / 3)]
train_idx = [0, n - 1]
train_x = x[train_idx]

model = SingleTaskGP(train_x, Tensor([0, 0]))

model.covar_module.base_kernel.lengthscale = 0.4
model.covar_module.outputscale = 1.0
model.likelihood.noise = 0.0001
model.eval()
y_prior = model(x)
torch.manual_seed(0)
y = y_prior.sample()

train_y = y[train_idx]
y_best = torch.max(train_y).item()
print(train_x)
print(train_y)
model.set_train_data(train_x, train_y, strict=False)
model.eval()
y_post = model(x)

f, ax = plt.subplots(2, 1, figsize=(6, 12))
with torch.no_grad():
    # Initialize plot

    # Get upper and lower confidence bounds
    lower, upper = y_post.confidence_region()
    # Plot training data as black stars
    ax[0].plot(x.squeeze().numpy(), y.numpy(), "r")
    ax[0].plot(train_x.squeeze().numpy(), train_y.numpy(), "k*")
    # Plot predictive means as blue line
    ax[0].plot(x.squeeze().numpy(), y_post.mean.detach().numpy(), "b")
    # Shade between the lower and upper confidence bounds
    ax[0].fill_between(x.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax[0].set_ylim([-3, 3])
    ax[0].legend(["True func", "Observed Data", "Mean", "Confidence"])

## compute EI
expected_improvement = ExpectedImprovement(model, best_f=y_best)

with torch.no_grad():
    ei_values = expected_improvement(x.unsqueeze(1))
    ax[1].plot(x.squeeze().numpy(), ei_values.numpy(), label="EI")

## compute two-step EI
two_step_ei = np.zeros((3, n))
times = np.zeros((3, n))
num_y_samples = 100
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
        bounds=Tensor([-bound, bound]).view(-1, 1),
        horizon=2,
        x_grid=x,
        idx=i,
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
    ax[1].plot(
        x.squeeze().numpy(),
        two_step_ei[1],
        label="two-step EI gauss-hermite %.2fs" % mean_time[1],
    )
    # ax[1].plot(x.squeeze().numpy(), two_step_ei[2], label='two-step EI adap-gauss %.2fs' % mean_time[2])
ax[1].legend()

print(times.mean(axis=1))
with open("rollout_test_results", "wb") as f:
    pickle.dump({"time": times, "ei": two_step_ei}, f)
plt.show()
