# How to run the code? 
To test the installation, run the following command in Mac or Linux terminal 

$ sh test_run_bo_task.sh

You can follow run_bo_task_example.ipynb to run an experiment in a Python
notebook, or use the following command to run in terminal 

$ python run_bo_task.py function acquisition num_iter_scale num_init_scale seed

Parameters:
* function: function name, e.g., branin, svm_on_grid (supported synthetic test function can be found in util/test_functions.py
* acquisition: acquisition method, e.g., EI, q.EI.pick (q="2", pick="best" or "sample")
* num_iter_scale: The number of further BO iterations would be "num_iter_scale*d", where d is the dimension of the function
* num_init_scale: The number of random initial observations would be "num_init_scale*d",
* seed: random seed for the generating the initial observations

for example, to run EI on branin:

$ python run_bo_task.py branin EI 20 2 1

to run our method:

$ python run_bo_task.py branin 2.EI.best 20 2 1

or

$ python run_bo_task.py svm_on_grid 4.EI.sample 20 2 1

By default, the results will be saved in './results',
or you can specify the save path:

$ python run_bo_task.py svm_on_grid 4.EI.sample 20 2 1 results_dir

you can also specify the log directory (default: ./log)

$ python run_bo_task.py svm_on_grid 4.EI.sample 20 2 1 results_dir log_dir

see also test_run_bo_task.sh


# Supported methods
In the following "q" means (pseudo) lookahead horizon. 

* "random": random policy used as baseline

* "EI": the vanilla expected improvement

* "q.EI.sample", "q.EI.best": BINOCULARS with batch size q, either "sample" or
choose the "best" from the optimal batch of size q.

* "q.rts.n": our implementation of the envelope-based "practical two-step EI" method (Wu &
        Frazier 2019); generalized to the case where the second step could have
batch size greater than 1, i.e., q-1; n is the number of y samples. e.g.,
      3.rts.10 means second step is of batch size 2, and 10 samples in approximating the expectation term in Eq. (4) of https://papers.nips.cc/paper/9174-practical-two-step-lookahead-bayesian-optimization.pdf

* "q.wsms.n": our proposed multi-step expected improvement, using warm-start for
optimizing the one-shot objective; "q.ms.n" means no warm-start. e.g., 2.wsms.1 for 2-step with one sample (i.e., 2-path)

* "q.wspms.n": our proposed pseudo multi-step method as in Eq. (7) of https://arxiv.org/pdf/2006.15779.pdf

* "q.rollout.n": our implementation of rollout
(https://papers.nips.cc/paper/6188-bayesian-optimization-with-a-finite-budget-an-approximate-dynamic-programming-approach.pdf)

* "q.glasses": our implementation of GLASSES
(https://arxiv.org/pdf/1510.06299.pdf)

See util/acq_func_optimization.py for details of the supported methods and more
parameterization. 

# Others
"./intuition_plots" contains the python notebook producing Figure 1 in
the BINOCULARS paper

