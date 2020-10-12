% load hyper_tuning_results.mat
% collabels = {'\svm', '\lda', 'LogReg', 'NN Boston', ...
%   'NN Cancer', 'Robot pushing 3d', 'Robot pushing 4d', 'Average'};
% rowlabels = {'random', '\ei', '\qei'};

denom = 1;
method_idx = [1 2 4 6 8 12 13:16];
paper_dir = '../../neurips2019/tables/';
print_gap_per_sec = 0;
add_median = 0;
for real = [0]
  if real == 0
    noncenter = 0;
    add_random = 1;
    method_idx = [1:(12+add_random)];
    method_idx = [1:11];
  else
    method_idx = 1:19; % [1 2 3 5  7 10 11];
    method_idx = 1:10;
  end
  for which_test = [1 2]
    if real
      %load(sprintf('real_all_gap%d_2ei_more_restarts.mat', denom));
      load(sprintf('real_all_gap%d.mat', denom));
      collabels = {'\svm', '\lda', 'LogReg', 'NN Boston', ...
        'NN Cancer', 'Robot pushing 3d', 'Robot pushing 4d'};
      filename = 'real_all_gap_better_qei_direct5_';
    else
%       load(sprintf('synthetic_all_functions_gap%d.mat', denom));
%       filename = 'synthetic_all_functions_gap_';
      load(sprintf('synthetic_gap_all_functions%d.mat', denom));
      load(sprintf('synthetic_gap_all_functions_rollout20d_glasses20d%d.mat', denom));

      collabels = cellstr(funcs);
%       filename = 'synthetic_gap_all_methods_';
      if noncenter
        filename = 'synthetic_gap_all_functions_rollout20d_glasses20d_noncenter_';
      else
        filename = 'synthetic_gap_all_functions_rollout20d_glasses20d_';
      end
    end
    format = '%.3f';
    if print_gap_per_sec
      gap = gap_per_sec;
      format = '%.3ef';
    end
    mean_gap = squeeze(nanmean(gap));
    if real
      func_idx = mean_gap(2,:) <= 1;   
    else
      func_idx = mean_gap(2,:) < 1.9;
      if noncenter
        func_idx = [1,3,7,8,9];
      end
    end
    collabels = collabels(func_idx);
    tests = {'ttest', 'signrank'};
    
    tex_filepath = fullfile(paper_dir, [filename tests{which_test} int2str(denom) '.tex']);
    collabels{end+1} = 'Average';
    if add_median
      collabels{end+1} = 'Median';
    end
    rowlabels = cellstr(methods);
    rowlabels = rowlabels(method_idx);
    nr = length(rowlabels);
    for i = 1:nr
      label = rowlabels{i};
      label = strrep(label, 'sample', 's');
      label = strrep(label, 'rollout', 'R');
      label = strrep(label, 'best', 'b');
      label = strrep(label, 'glasses.20', 'G');
      label = strrep(label, 'glasses.0', 'G');
      label = strrep(label, '.initL', '');
      label = strrep(label, 'random', 'Rand');
      rowlabels{i} = label;
    end
    gap = gap(:,method_idx, :);
    gap = gap(:,:,func_idx);
    
    matrix2latex_highlight_best(gap, tex_filepath, which_test, ...
      .05, [], 1, 0, 1, add_median,...
      'columnLabels', rowlabels, ...
      'rowLabels', collabels, 'format', format, 'fopen_mode', 'w')
  end
end
% two_ei = squeeze(gap(:,10, :));
% two_r  = squeeze(gap(:,end-1,:));
% [p, h] = signrank(two_ei(:), two_r(:), 'tail', 'right')
% 
% two_ei = squeeze(gap(:,1, 4));
% two_r  = squeeze(gap(:,3,4));
% [p, h] = signrank(two_ei(:), two_r(:), 'tail', 'right')