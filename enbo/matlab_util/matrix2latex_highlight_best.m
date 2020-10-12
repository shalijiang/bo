function matrix2latex_highlight_best(table, filename, which_test, pvalue_level, ...
  exclude_row_idx, transpose, small_better, add_average, add_median, varargin)
% this function print a matrix to latex table
% table: s by m by n
% first take mean of table get m by n matrix
% find the highest value of each column
% conduct paired t-test for the remaining row against the best
% highlight the best and all that are not significantly worse

[rpt, m, n] = size(table);
mean_table = squeeze(nanmean(table));
compare_idx = 1:m;
%compare_idx(exclude_row_idx) = [];
if small_better
  [~, I] = min(mean_table(compare_idx,:));
else
  [~, I] = max(mean_table(compare_idx,:));
end
I = compare_idx(I);

highlight = zeros(m, n);
highlight_best = zeros(m,n);
tail = 'left';
if small_better
  tail = 'right';
end
for i = 1:n
  best = I(i);
  highlight_best(best, i) = 1;
  for j = 1:m
    if j == best, continue; end
    if which_test == 1  % which_test = 1 for ttest, 2 for signrank test
      [h, pvalue]=ttest(table(:,j,i), table(:,best,i), 'Tail', tail);
    else
      [pvalue, h]=signrank(table(:,j,i), table(:,best,i), 'Tail', tail);
    end
    if isnan(pvalue) || pvalue >= pvalue_level
      highlight(j, i) = 1;
      fprintf('ttest %d vs %d: bs %d: h=%d pvalue=%f\n', ...
        j, best, i, h, pvalue);
    end
  end
end

if add_average
  table1 = permute(table, [1 3 2]);
  table2 = reshape(table1, [rpt*n m]);
  mean_table2 = nanmean(table2);
  if small_better
    [~, I] = min(mean_table2);
  else
    [~, I] = max(mean_table2);
  end
  best = I;
  highlight_best(best, end+1) = 1;
  highlight(:, end+1) = 0;
  % if pirnt median
  if add_median
    highlight_best(best, end+1) = 1;
    highlight(:, end+1) = 0;
  end
  for j = 1:m
    if j == best, continue; end
    if which_test == 1  % which_test = 1 for ttest, 2 for signrank test
      [h, pvalue]=ttest(table2(:,j), table2(:,best), 'Tail', tail);
    else
      [pvalue, h]=signrank(table2(:,j), table2(:,best), 'Tail', tail);
    end
    if isnan(pvalue) || pvalue >= pvalue_level
      highlight(j, end) = 1;
      if add_median
        highlight(j, end-1) = 1;
      end
      fprintf('aggregate: ttest %d vs %d: bs %d: h=%d pvalue=%f\n', ...
        j, best, i, h, pvalue);
    end
  end
  mean_table(:, end+1) = mean_table2';
  if add_median
    mean_table(:, end+1) = nanmedian(table2)';
  end
end

if transpose
  mean_table = mean_table';
  highlight_best = highlight_best';
  highlight = highlight';
end

matrix2latex_std(mean_table, filename, varargin{:}, ...
  'best_bold', highlight_best, 'highlight_same_as_best', highlight);
