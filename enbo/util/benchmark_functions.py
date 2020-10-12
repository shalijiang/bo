"""
Implementation of optimization benchmark function using Pytorch
reference: https://www.sfu.ca/~ssurjano/optimization.html
"""
import torch
from torch import Tensor
import math
import numpy as np
from .utils import DataTransformer, standardize


class BenchmarkFunction(object):
    def __init__(self, noise_std=None, maximize=False):
        self.noise_std = noise_std
        self.maximize = maximize
        self.is_grid = False
        self.bounds = None
        self.dim = 1

    @property
    def optimal_value(self):
        return self._optimal_value if not self.maximize else -self._optimal_value

    def evaluate_true(self, x):
        raise NotImplementedError()

    def evaluate(self, x, transformer=None):
        if transformer is not None:
            # transform from some other space to the original domain
            x = transformer(x)

        batch = x.ndimension() > 1
        x = x if batch else x.unsqueeze(0)
        f = self.evaluate_true(x)
        if self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.maximize:
            f = -f
        return f if batch else f.squeeze(0)

    def discretize_domain(self, num_sobol_samples_scale=1000):
        self.is_grid = True
        self.num_sobol_samples = num_sobol_samples_scale * self.dim
        soboleng = torch.quasirandom.SobolEngine(dimension=self.dim)
        self.x = soboleng.draw(self.num_sobol_samples)
        transformer = DataTransformer(self.bounds)
        self.raw_x = transformer.from_normalized_to_original(self.x)
        self.raw_y = self.evaluate(self.raw_x)
        self.y = standardize(self.raw_y)


class Ackley(BenchmarkFunction):
    def __init__(
        self, dim=2, noise_std=None, maximize=False, a=20.0, b=0.2, c=2 * math.pi
    ):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = dim
        self.bounds = Tensor([[-32.768, 32.768] for _ in range(dim)]).t()
        self._optimal_value = 0.0
        self.x_star = [[0.0] * dim]
        self.a = a
        self.b = b
        self.c = c

    def evaluate_true(self, x):
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b * torch.sqrt(torch.mean(x ** 2, dim=1)))
        part2 = -torch.exp(torch.mean(torch.cos(c * x), dim=1))
        return part1 + part2 + a + math.e


class Beale(BenchmarkFunction):
    def __init__(self, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 2
        self.bounds = Tensor([[-4.5, 4.5], [-4.5, 4.5]]).t()
        self._optimal_value = 0.0
        self.x_star = [[3.0, 0.5]]

    def evaluate_true(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        part1 = (1.5 - x1 + x1 * x2) ** 2
        part2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        part3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
        return part1 + part2 + part3


class Branin(BenchmarkFunction):
    def __init__(self, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 2
        self.bounds = Tensor([[-5.0, 10.0], [0.0, 15.0]]).t()
        self._optimal_value = 0.397887
        self.x_star = [[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]]

    def evaluate_true(self, x):
        t1 = (
            x[:, 1]
            - 5.1 / (4 * math.pi ** 2) * x[:, 0] ** 2
            + 5 / math.pi * x[:, 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(x[:, 0])
        return t1 ** 2 + t2 + 10


class Bukin(BenchmarkFunction):
    def __init__(self, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 2
        self.bounds = Tensor([[-15.0, -5.0], [-3.0, 3.0]]).t()
        self._optimal_value = 0.0
        self.x_star = [[-10.0, 1.0]]

    def evaluate_true(self, x):
        part1 = 100.0 * torch.sqrt(torch.abs(x[:, 1] - 0.01 * x[:, 0] ** 2))
        part2 = 0.01 * torch.abs(x[:, 0] + 10.0)
        return part1 + part2


class CrossInTray(BenchmarkFunction):
    def __init__(self, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 2
        self.bounds = Tensor([[-10.0, 10.0], [-10.0, 10.0]]).t()
        self._optimal_value = -2.06261
        self.x_star = [
            [1.3491, -1.3491],
            [1.3491, 1.3491],
            [-1.3491, 1.3491],
            [-1.3491, -1.3491],
        ]

    def evaluate_true(self, x):
        # torch.exp(100) = Inf if use single precision float
        x = x.type(torch.DoubleTensor)
        x1 = x[:, 0]
        x2 = x[:, 1]
        result = -0.0001 * (
            torch.abs(
                torch.sin(x1)
                * torch.sin(x2)
                * torch.exp(torch.abs(100.0 - torch.sqrt(x1 ** 2 + x2 ** 2) / math.pi))
            )
            + 1.0
        ).pow(0.1)

        return result.type(torch.float)


class DropWave(BenchmarkFunction):
    def __init__(self, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 2
        self.bounds = Tensor([[-5.12, 5.12], [-5.12, 5.12]]).t()
        self._optimal_value = -1.0
        self.x_star = [[0.0, 0.0]]

    def evaluate_true(self, x):
        sum2 = torch.sum(x ** 2, dim=1)
        part1 = 1.0 + torch.cos(12.0 * torch.sqrt(sum2))
        part2 = 0.5 * sum2 + 2.0
        return -part1 / part2


class DixonPrice(BenchmarkFunction):
    def __init__(self, dim=2, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = dim
        self.bounds = Tensor([[-10.0, 10.0] for _ in range(dim)]).t()
        self._optimal_value = 0.0
        self.x_star = [
            [math.pow(2.0, -(1.0 - 2.0 ** (-(i - 1)))) for i in range(1, dim + 1)]
        ]

    def evaluate_true(self, x):
        d = self.dim
        part1 = (x[:, 0] - 1) ** 2
        i = x.new(range(2, d + 1))
        part2 = torch.sum(i * (2.0 * x[:, 1:] ** 2 - x[:, :-1]) ** 2, dim=1)
        return part1 + part2


class EggHolder(BenchmarkFunction):
    def __init__(self, noise_std=None, maximize=False, a=20.0, b=0.2, c=2 * math.pi):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 2
        self.bounds = Tensor([[-512.0, 512.0], [-512.0, 512.0]]).t()
        self._optimal_value = -959.6407
        self.x_star = [[512.0, 404.2319]]

    def evaluate_true(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        part1 = -(x2 + 47.0) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2.0 + 47.0)))
        part2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47.0))))
        return part1 + part2


class Griewank(BenchmarkFunction):
    def __init__(self, dim=2, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = dim
        self.bounds = Tensor([[-600.0, 600.0] for _ in range(dim)]).t()
        self._optimal_value = 0.0
        self.x_star = [[0.0] * dim]

    def evaluate_true(self, x):
        part1 = torch.sum(x ** 2 / 4000.0, dim=1)
        d = x.shape[1]
        part2 = -torch.prod(
            torch.cos(x / torch.sqrt(x.new(range(1, d + 1))).view(1, -1))
        )
        return part1 + part2 + 1.0


class GrLee12(BenchmarkFunction):
    def __init__(self, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 1
        self.bounds = Tensor([[0.5, 2.5]]).t()
        self._optimal_value = None
        self.x_star = None

    def evaluate_true(self, x):
        return (torch.sin(10.0 * math.pi * x) / (2.0 * x) + (x - 1.0) ** 4).squeeze()


class Hartmann(BenchmarkFunction):
    def __init__(self, dim=3, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = dim
        self.bounds = Tensor([[0.0, 1.0] for _ in range(self.dim)]).t()
        self._optimal_value = None
        self.x_star = None
        self.ALPHA = [1.0, 1.2, 3.0, 3.2]
        if dim == 3:
            self.A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            self.P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
            self._optimal_value = -3.86278
            self.x_star = [[0.114614, 0.555649, 0.852547]]
        elif dim == 4:
            self.A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            self.P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dim == 6:
            self.A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            self.P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
            self._optimal_value = -3.32237
            self.x_star = [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]
        else:
            raise ValueError("Hartmann with dim %d not defined" % dim)

    def evaluate_true(self, x):
        inner_sum = torch.sum(
            x.new(self.A) * (x.unsqueeze(1) - 0.0001 * x.new(self.P)) ** 2, dim=2
        )
        H = -torch.sum(x.new(self.ALPHA) * torch.exp(-inner_sum), dim=1)
        if self.dim == 4:
            H = (1.1 + H) / 0.839
        return H


class HolderTable(BenchmarkFunction):
    def __init__(self, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 2
        self.bounds = Tensor([[-10.0, 10.0] for _ in range(self.dim)]).t()
        self._optimal_value = -19.2085
        self.x_star = [
            [8.05502, 9.66459],
            [-8.05502, -9.66459],
            [-8.05502, 9.66459],
            [8.05502, -9.66459],
        ]

    def evaluate_true(self, x):
        term = torch.abs(1 - torch.norm(x, dim=1) / math.pi)
        return -torch.abs(torch.sin(x[:, 0]) * torch.cos(x[:, 1]) * torch.exp(term))


class Levy(BenchmarkFunction):
    def __init__(self, dim=2, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = dim
        self.bounds = Tensor([[-10.0, 10.0] for _ in range(dim)]).t()
        self._optimal_value = 0.0
        self.x_star = [[1.0] * dim]

    def evaluate_true(self, x):
        w = 1.0 + (x - 1.0) / 4.0
        part1 = torch.sin(math.pi * w[:, 0]) ** 2
        part2 = torch.sum(
            (w[:, :-1] - 1.0) ** 2
            * (1.0 + 10.0 * torch.sin(math.pi * w[:, :-1] + 1.0) ** 2),
            dim=1,
        )
        part3 = (w[:, -1] - 1.0) ** 2 * (1.0 + torch.sin(2.0 * math.pi * w[:, -1]) ** 2)
        return part1 + part2 + part3


class Michalewicz(BenchmarkFunction):
    def __init__(self, dim=2, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = dim
        self.bounds = Tensor([[0.0, math.pi] for _ in range(dim)]).t()
        self.x_star = None
        if dim == 2:
            self._optimal_value = -1.8013
            self.x_star = [[2.20, 1.57]]
        elif dim == 5:
            self._optimal_value = -4.687658
        elif dim == 10:
            self._optimal_value = -9.66015

    def evaluate_true(self, x):
        i = x.new(range(1, self.dim + 1))
        m = 10
        return -torch.sum(
            torch.sin(x) * torch.sin(i * x ** 2 / math.pi) ** (2 * m), dim=1
        )


class Powell(BenchmarkFunction):
    def __init__(self, dim=4, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        assert dim % 4 == 0, "Powell dim must mutiple of 4"
        self.dim = dim
        self.bounds = Tensor([[-4.0, 5.0] for _ in range(dim)]).t()
        self._optimal_value = 0.0
        self.x_star = [[0.0] * dim]

    def evaluate_true(self, x):
        result = torch.zeros_like(x[:, 0])
        for i in range(self.dim // 4):
            i_ = i + 1
            part1 = (x[:, 4 * i_ - 4] + 10.0 * x[:, 4 * i_ - 3]) ** 2
            part2 = 5.0 * (x[:, 4 * i_ - 2] - x[:, 4 * i_ - 1]) ** 2
            part3 = (x[:, 4 * i_ - 3] - 2.0 * x[:, 4 * i_ - 2]) ** 4
            part4 = 10.0 * (x[:, 4 * i_ - 4] - x[:, 4 * i_ - 1]) ** 4
            result += part1 + part2 + part3 + part4
        return result


class Rastrigin(BenchmarkFunction):
    def __init__(self, dim=2, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = dim
        self.bounds = Tensor([[-5.12, 5.12] for _ in range(dim)]).t()
        self._optimal_value = 0.0
        self.x_star = [[0.0] * dim]

    def evaluate_true(self, x):
        d = x.shape[1]
        return 10.0 * d + torch.sum(x ** 2 - 10.0 * torch.cos(2.0 * math.pi * x), dim=1)


class Rosenbrock(BenchmarkFunction):
    def __init__(self, dim=2, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = dim
        self.bounds = Tensor([[-2.048, 2.048] for _ in range(dim)]).t()
        self._optimal_value = 0.0
        self.x_star = [[1.0] * dim]

    def evaluate_true(self, x):
        return torch.sum(
            100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2, dim=1
        )


class Shubert(BenchmarkFunction):
    def __init__(self, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 2
        self.bounds = Tensor([[-5.12, 5.12], [-5.12, 5.12]]).t()
        self._optimal_value = -186.7309
        self.x_star = None

    def evaluate_true(self, x):
        n = x.shape[0]
        i = x.new(range(1, 6)).repeat(n, 1)
        i_plus_1 = x.new(range(2, 7)).repeat(n, 1)
        part1 = torch.sum(i * torch.cos(i_plus_1 * x[:, 0].view(-1, 1) + i), dim=1)
        part2 = torch.sum(i * torch.cos(i_plus_1 * x[:, 1].view(-1, 1) + i), dim=1)
        return part1 * part2


class Shekel(BenchmarkFunction):
    def __init__(self, m=10, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 4
        self.m = m
        self.bounds = Tensor([[0.0, 10.0] for _ in range(self.dim)]).t()
        if m == 5:
            self._optimal_value = -10.1532
        elif m == 7:
            self._optimal_value = -10.4029
        elif m == 10:
            self._optimal_value = -10.5364
        self.beta = [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]
        self.C = [
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
        ]
        self.x_star = [[4, 4, 4, 4]]

    def evaluate_true(self, x):
        C = x.new(self.C).t()
        beta = x.new(self.beta) / 10.0
        result = 0.0
        for i in range(self.m):
            result += 1.0 / (torch.sum((x - C[i]) ** 2, dim=1) + beta[i])
        return -result


class SixHumpCamel(BenchmarkFunction):
    def __init__(self, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 2
        self.bounds = Tensor([[-3.0, 3.0], [-2.0, 2.0]]).t()
        self._optimal_value = -1.0316
        self.x_star = [[0.0898, -0.7126], [-0.0898, 0.7126]]

    def evaluate_true(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return (
            (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2
            + x1 * x2
            + (4 * x2 ** 2 - 4) * x2 ** 2
        )


class StybTang(BenchmarkFunction):
    def __init__(self, dim=2, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = dim
        self.bounds = Tensor([[-5.0, 5.0] for _ in range(self.dim)]).t()
        self._optimal_value = -39.166166 * dim
        self.x_star = [[-2.903534] * dim]

    def evaluate_true(self, x):
        return 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x).sum(dim=1)


class ThreeHumpCamel(BenchmarkFunction):
    def __init__(self, noise_std=None, maximize=False):
        super().__init__(noise_std=noise_std, maximize=maximize)
        self.dim = 2
        self.bounds = Tensor([[-5.0, 5.0], [-5.0, 5.0]]).t()
        self._optimal_value = 0.0
        self.x_star = [[0.0, 0.0]]

    def evaluate_true(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return 2.0 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 6.0 + x1 * x2 + x2 ** 2
