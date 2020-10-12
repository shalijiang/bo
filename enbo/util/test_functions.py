from . import benchmark_functions
from .global_variables import MAXIMIZE

benchmark_funcs = {
    "branin": benchmark_functions.Branin(maximize=MAXIMIZE),
    "rosenbrock2": benchmark_functions.Rosenbrock(dim=2, maximize=MAXIMIZE),
    "rosenbrock4": benchmark_functions.Rosenbrock(dim=4, maximize=MAXIMIZE),
    "rosenbrock6": benchmark_functions.Rosenbrock(dim=6, maximize=MAXIMIZE),
    "hartmann3": benchmark_functions.Hartmann(dim=3, maximize=MAXIMIZE),
    "hartmann6": benchmark_functions.Hartmann(dim=6, maximize=MAXIMIZE),
    "eggholder": benchmark_functions.EggHolder(maximize=MAXIMIZE),
    "dropwave": benchmark_functions.DropWave(maximize=MAXIMIZE),
    "beale": benchmark_functions.Beale(maximize=MAXIMIZE),
    "shubert": benchmark_functions.Shubert(maximize=MAXIMIZE),
    "sixhumpcamel6": benchmark_functions.SixHumpCamel(maximize=MAXIMIZE),
    "holder": benchmark_functions.HolderTable(maximize=MAXIMIZE),
    "rosenbrock10": benchmark_functions.Rosenbrock(dim=10, maximize=MAXIMIZE),
    "threehumpcamel": benchmark_functions.ThreeHumpCamel(maximize=MAXIMIZE),
    "rastrigin2": benchmark_functions.Rastrigin(dim=2, maximize=MAXIMIZE),
    "rastrigin4": benchmark_functions.Rastrigin(dim=4, maximize=MAXIMIZE),
    "rastrigin6": benchmark_functions.Rastrigin(dim=6, maximize=MAXIMIZE),
    "ackley2": benchmark_functions.Ackley(dim=2, maximize=MAXIMIZE),
    "ackley5": benchmark_functions.Ackley(dim=5, maximize=MAXIMIZE),
    "levy2": benchmark_functions.Levy(dim=2, maximize=MAXIMIZE),
    "levy3": benchmark_functions.Levy(dim=3, maximize=MAXIMIZE),
    "levy4": benchmark_functions.Levy(dim=4, maximize=MAXIMIZE),
    "levy10": benchmark_functions.Levy(dim=10, maximize=MAXIMIZE),
    "griewank2": benchmark_functions.Griewank(dim=2, maximize=MAXIMIZE),
    "griewank5": benchmark_functions.Griewank(dim=5, maximize=MAXIMIZE),
    "stybtang2": benchmark_functions.StybTang(dim=2, maximize=MAXIMIZE),
    "stybtang4": benchmark_functions.StybTang(dim=4, maximize=MAXIMIZE),
    "stybtang6": benchmark_functions.StybTang(dim=6, maximize=MAXIMIZE),
    "stybtang8": benchmark_functions.StybTang(dim=8, maximize=MAXIMIZE),
    "powell4": benchmark_functions.Powell(dim=4, maximize=MAXIMIZE),
    "dixonprice2": benchmark_functions.DixonPrice(dim=2, maximize=MAXIMIZE),
    "dixonprice4": benchmark_functions.DixonPrice(dim=4, maximize=MAXIMIZE),
    "crossintray": benchmark_functions.CrossInTray(maximize=MAXIMIZE),
    "bukin": benchmark_functions.Bukin(maximize=MAXIMIZE),
    "shekel5": benchmark_functions.Shekel(m=5, maximize=MAXIMIZE),
    "shekel7": benchmark_functions.Shekel(m=7, maximize=MAXIMIZE),
    "shekel10": benchmark_functions.Shekel(m=10, maximize=MAXIMIZE),
    "michal2": benchmark_functions.Michalewicz(dim=2, maximize=MAXIMIZE),
    "michal5": benchmark_functions.Michalewicz(dim=5, maximize=MAXIMIZE),
    "michal10": benchmark_functions.Michalewicz(dim=10, maximize=MAXIMIZE),
}

func_name_list = [
    "branin",
    "rosenbrock2",
    "rosenbrock4",
    "rosenbrock6",
    "hartmann3",
    "hartmann6",
    "eggholder",
    "dropwave",
    "beale",
    "shubert",
    "sixhumpcamel6",
    "holder",
    "rosenbrock10",
    "threehumpcamel",
    "rastrigin2",
    "rastrigin4",
    "rastrigin6",
    "ackley2",
    #'ackley5',
    # 'levy2',
    # 'levy3',
    # 'levy4',
    # 'griewank2',
    "griewank5",
    "stybtang2",
    "stybtang4",
    "stybtang6",
    # 'stybtang8',
    # 'powell2',
    # 'powell4',
    # 'dixonprice2',
    # 'dixonprice4',
    # 'crossintray',
    # 'bukin',
    "shekel5",
    "shekel7",
    "shekel10",
    "michal2",
    "michal5",
    "michal10",
    # 'svm',
    # 'lda',
    # 'logreg',
    # 'nn_boston',
    # 'nn_cancer',
    # 'robot3',
    # 'robot4',
    # 'cosmo',
]
