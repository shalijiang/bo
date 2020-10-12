# Changelog

The release log for BoTorch.

## [0.2.1] - Jan 15, 2020

Minor bug fix release.

#### New Features
* Add a static method for getting batch shapes for batched MO models (#346).

#### Bug fixes
* Revamp qKG constructor to avoid issue with missing objective (#351).
* Make sure MVES can support sampled costs like KG (#352).

#### Other changes
* Allow custom module-to-array handling in fit_gpytorch_scipy (#341).


## [0.2.0] - Dec 20, 2019

Max-value entropy acquisition function, cost-aware / multi-fidelity optimization,
subsetting models, outcome transforms.

#### Compatibility
* Require PyTorch >=1.3.1 (#313).
* Require GPyTorch >=1.0 (#342).

#### New Features
* Add cost-aware KnowledgeGradient (`qMultiFidelityKnowledgeGradient`) for
  multi-fidelity optimization (#292).
* Add `qMaxValueEntropy` and `qMultiFidelityMaxValueEntropy` max-value entropy
  search acquisition functions (#298).
* Add `subset_output` functionality to (most) models (#324).
* Add outcome transforms and input transforms (#321).
* Add `outcome_transform` kwarg to model constructors for automatic outcome
  transformation and un-transformation (#327).
* Add cost-aware utilities for cost-sensitive acquisiiton functions (#289).
* Add `DeterminsticModel` and `DetermisticPosterior` abstractions (#288).
* Add `AffineFidelityCostModel` (f838eacb4258f570c3086d7cbd9aa3cf9ce67904).
* Add `project_to_target_fidelity` and `expand_trace_observations` utilties for
  use in multi-fidelity optimization (1ca12ac0736e39939fff650cae617680c1a16933).

#### Performance Improvements
* New `prune_baseline` option for pruning `X_baseline` in
  `qNoisyExpectedImprovement` (#287).
* Do not use approximate MLL computation for deterministic fitting (#314).
* Avoid re-evaluating the acquisition function in `gen_candidates_torch` (#319).
* Use CPU where possible in `gen_batch_initial_conditions` to avoid memory
  issues on the GPU (#323).

#### Bug fixes
* Properly register `NoiseModelAddedLossTerm` in `HeteroskedasticSingleTaskGP`
  (671c93a203b03ef03592ce322209fc5e71f23a74).
* Fix batch mode for `MultiTaskGPyTorchModel` (#316).
* Honor `propagate_grads` argument in `fantasize` of `FixedNoiseGP` (#303).
* Properly handle `diag` arg in `LinearTruncatedFidelityKernel` (#320).

#### Other changes
* Consolidate and simplify multi-fidelity models (#308).
* New license header style (#309).
* Validate shape of `best_f` in `qExpectedImprovement` (#299).
* Support specifying observation noise explicitly for all models (#256).
* Add `num_outputs` property to the `Model` API (#330).
* Validate output shape of models upon instantiating acquisition functions (#331).

#### Tests
* Silence warnings outside of explicit tests (#290).
* Enforce full sphinx docs coverage in CI (#294).


## [0.1.4] - Oct 1, 2019

Knowledge Gradient acquisition function (one-shot), various maintenance

#### Breaking Changes
* Require explicit output dimensions in BoTorch models (#238)
* Make `joint_optimize` / `sequential_optimize` return acquisition function
  values (#149) [note deprecation notice below]
* `standardize` now works on the second to last dimension (#263)
* Refactor synthetic test functions (#273)

#### New Features
* Add `qKnowledgeGradient` acquisition function (#272, #276)
* Add input scaling check to standard models (#267)
* Add `cyclic_optimize`, convergence criterion class (#269)
* Add `settings.debug` context manager (#242)

#### Deprecations
* Consolidate `sequential_optimize` and `joint_optimize` into `optimize_acqf`
  (#150)

#### Bug fixes
* Properly pass noise levels to GPs using a `FixedNoiseGaussianLikelihood` (#241)
  [requires gpytorch > 0.3.5]
* Fix q-batch dimension issue in `ConstrainedExpectedImprovement`
  (6c067185f56d3a244c4093393b8a97388fb1c0b3)
* Fix parameter constraint issues on GPU (#260)

#### Minor changes
* Add decorator for concatenating pending points (#240)
* Draw independent sample from prior for each hyperparameter (#244)
* Allow `dim > 1111` for `gen_batch_initial_conditions` (#249)
* Allow `optimize_acqf` to use `q>1` for `AnalyticAcquisitionFunction` (#257)
* Allow excluding parameters in fit functions (#259)
* Track the final iteration objective value in `fit_gpytorch_scipy` (#258)
* Error out on unexpected dims in parameter constraint generation (#270)
* Compute acquisition values in gen_ functions w/o grad (#274)

#### Tests
* Introduce BotorchTestCase to simplify test code (#243)
* Refactor tests to have monolithic cuda tests (#261)


## [0.1.3] - Aug 9, 2019

Compatibility & maintenance release

#### Compatibility
* Updates to support breaking changes in PyTorch to boolean masks and tensor
  comparisons (#224).
* Require PyTorch >=1.2 (#225).
* Require GPyTorch >=0.3.5 (itself a compatibility release).

#### New Features
* Add `FixedFeatureAcquisitionFunction` wrapper that simplifies optimizing
  acquisition functions over a subset of input features (#219).
* Add `ScalarizedObjective` for scalarizing posteriors (#210).
* Change default optimization behavior to use L-BFGS-B by for box constraints
  (#207).

#### Bug fixes
* Add validation to candidate generation (#213), making sure constraints are
  strictly satisfied (rater than just up to numerical accuracy of the optimizer).

#### Minor changes
* Introduce `AcquisitionObjective` base class (#220).
* Add propagate_grads context manager, replacing the `propagate_grads` kwarg in
  model `posterior()` calls (#221)
* Add `batch_initial_conditions` argument to `joint_optimize()` for
  warm-starting the optimization (ec3365a37ed02319e0d2bb9bea03aee89b7d9caa).
* Add `return_best_only` argument to `joint_optimize()` (#216). Useful for
  implementing advanced warm-starting procedures.


## [0.1.2] - July 9, 2019

Maintenance release

#### Bug fixes
* Avoid [PyTorch bug]((https://github.com/pytorch/pytorch/issues/22353)
  resulting in bad gradients on GPU by requiring GPyTorch >= 0.3.4
* Fixes to resampling behavior in MCSamplers (#204)

#### Experimental Features
* Linear truncated kernel for multi-fidelity bayesian optimization (#192)
* SingleTaskMultiFidelityGP for GP models that have fidelity parameters (#181)


## [0.1.1] - June 27, 2019

API updates, more robust model fitting

#### Breaking changes
* rename `botorch.qmc` to `botorch.sampling`, move MC samplers from
  `acquisition.sampler` to `botorch.sampling.samplers` (#172)

#### New Features
* Add `condition_on_observations` and `fantasize` to the Model level API (#173)
* Support pending observations generically for all `MCAcqusitionFunctions` (#176)
* Add fidelity kernel for training iterations/training data points (#178)
* Support for optimization constraints across `q`-batches (to support things like
  sample budget constraints) (2a95a6c3f80e751d5cf8bc7240ca9f5b1529ec5b)
* Add ModelList <-> Batched Model converter (#187)
* New test functions
    * basic: `neg_ackley`, `cosine8`, `neg_levy`, `neg_rosenbrock`, `neg_shekel`
      (e26dc7576c7bf5fa2ba4cb8fbcf45849b95d324b)
    * for multi-fidelity BO: `neg_aug_branin`, `neg_aug_hartmann6`,
      `neg_aug_rosenbrock` (ec4aca744f65ca19847dc368f9fee4cc297533da)

#### Improved functionality:
* More robust model fitting
    * Catch gpytorch numerical issues and return `NaN` to the optimizer (#184)
    * Restart optimization upon failure by sampling hyperparameters from their prior (#188)
    * Sequentially fit batched and `ModelListGP` models by default (#189)
    * Change minimum inferred noise level (e2c64fef1e76d526a33951c5eb75ac38d5581257)
* Introduce optional batch limit in `joint_optimize` to increases scalability of
  parallel optimization (baab5786e8eaec02d37a511df04442471c632f8a)
* Change constructor of `ModelListGP` to comply with GPyTorch’s `IndependentModelList`
  constructor (a6cf739e769c75319a67c7525a023ece8806b15d)
* Use `torch.random` to set default seed for samplers (rather than `random`) to
  making sampling reproducible when setting `torch.manual_seed`
  (ae507ad97255d35f02c878f50ba68a2e27017815)

####  Performance Improvements
* Use `einsum` in `LinearMCObjective` (22ca29535717cda0fcf7493a43bdf3dda324c22d)
* Change default Sobol sample size for `MCAquisitionFunctions` to be base-2 for
  better MC integration performance (5d8e81866a23d6bfe4158f8c9b30ea14dd82e032)
* Add ability to fit models in `SumMarginalLogLikelihood` sequentially (and make
  that the default setting) (#183)
* Do not construct the full covariance matrix when computing posterior of
  single-output BatchedMultiOutputGPyTorchModel (#185)

#### Bug fixes
* Properly handle observation_noise kwarg for BatchedMultiOutputGPyTorchModels (#182)
* Fix a issue where `f_best` was always max for NoisyExpectedImprovement
  (de8544a75b58873c449b41840a335f6732754c77)
* Fix bug and numerical issues in `initialize_q_batch`
  (844dcd1dc8f418ae42639e211c6bb8e31a75d8bf)
* Fix numerical issues with `inv_transform` for qMC sampling (#162)

#### Other
* Bump GPyTorch minimum requirement to 0.3.3



## [0.1.0] - April 30, 2019

First public beta release.
