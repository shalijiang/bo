% should load an table of size (num_repeats, num_methods, num_functions)
load('hyper_tuning_results.mat');  

collabels = {'\svm', '\lda', 'LogReg', 'NN Boston', ...
  'NN Cancer', 'Robot pushing 3d', 'Robot pushing 4d', 'Average'};
rowlabels = {'random', '\ei', '\qei'};
which_test = 1; % 1 for ttest, 2 for sign rank test
tex_filepath = 'tmp.tex';
matrix2latex_highlight_best(gap, tex_filepath, which_test, .05, [], 1, 0, 1, ...
  'columnLabels', rowlabels, ...
  'rowLabels', collabels, 'format', '%.3f', 'fopen_mode', 'w')
