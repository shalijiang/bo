function matrix2latex_std(matrix, filename, varargin)

% function: matrix2latex(...)
% Author:   M. Koehler
% Contact:  koehler@in.tum.de
% Version:  1.1
% Date:     May 09, 2004

% This software is published under the GNU GPL, by the free software
% foundation. For further reading see: http://www.gnu.org/licenses/licenses.html#GPL

% Usage:
% matrix2late(matrix, filename, varargs)
% where
%   - matrix is a 2 dimensional numerical or cell array
%   - filename is a valid filename, in which the resulting latex code will
%   be stored
%   - varargs is one ore more of the following (denominator, value) combinations
%      + 'rowLabels', array -> Can be used to label the rows of the
%      resulting latex table
%      + 'columnLabels', array -> Can be used to label the columns of the
%      resulting latex table
%      + 'alignment', 'value' -> Can be used to specify the alginment of
%      the table within the latex document. Valid arguments are: 'l', 'c',
%      and 'r' for left, center, and right, respectively
%      + 'format', 'value' -> Can be used to format the input data. 'value'
%      has to be a valid format string, similar to the ones used in
%      fprintf('format', value);
%      + 'size', 'value' -> One of latex' recognized font-sizes, e.g. tiny,
%      HUGE, Large, large, LARGE, etc.
%
% Example input:
%   matrix = [1.5 1.764; 3.523 0.2];
%   rowLabels = {'row 1', 'row 2'};
%   columnLabels = {'col 1', 'col 2'};
%   matrix2latex(matrix, 'out.tex', 'rowLabels', rowLabels, 'columnLabels', columnLabels, 'alignment', 'c', 'format', '%-6.2f', 'size', 'tiny');
%
% The resulting latex file can be included into any latex document by:
% /input{out.tex}
%
% Enjoy life!!!

rowLabels = [];
colLabels = [];
alignment = 'l';
format = [];
textsize = [];
fopen_mode = 'a';
bold_entries = [];
highlight_entries = [];
highlight_color = 'blue';
hlines = [];
textit = true;
textcolor = true;
exclude_row_idx = [];
% if (rem(nargin,2) == 1 || nargin < 2)
%   error('matrix2latex: ', 'Incorrect number of arguments to %s.', mfilename);
% end

okargs = {'rowlabels','columnlabels', 'alignment', 'format', 'size', ...
  'fopen_mode', 'best_bold', 'highlight_same_as_best', 'highlight_color', ...
  'hlines'};
for j=1:2:(nargin-2)
  pname = lower(varargin{j});
  pval = varargin{j+1};
  k = strmatch(lower(pname), okargs);
  if isempty(k)
    error('matrix2latex: ', 'Unknown parameter name: %s.', pname);
  elseif length(k)>1
    error('matrix2latex: ', 'Ambiguous parameter name: %s.', pname);
  else
    switch(k)
      case 1  % rowlabels
        rowLabels = pval;
        if isnumeric(rowLabels)
          rowLabels = cellstr(num2str(rowLabels(:)));
        end
      case 2  % column labels
        colLabels = pval;
        if isnumeric(colLabels)
          colLabels = cellstr(num2str(colLabels(:)));
        end
      case 3  % alignment
        alignment = lower(pval);
        if alignment == 'right'
          alignment = 'r';
        end
        if alignment == 'left'
          alignment = 'l';
        end
        if alignment == 'center'
          alignment = 'c';
        end
        if alignment ~= 'l' && alignment ~= 'c' && alignment ~= 'r'
          alignment = 'l';
          warning('matrix2latex: ', 'Unkown alignment. (Set it to \''left\''.)');
        end
      case 4  % format
        format = lower(pval);
      case 5  % format
        textsize = pval;
      case 6
        fopen_mode = pval;
      case 7
        bold_entries = pval;
      case 8
        highlight_entries = pval;
      case 9
        highlight_color = pval;
      case 10
        hlines = pval;
    end
  end
end

fid = fopen(filename, fopen_mode);
fprintf(fid, '\n');

width = size(matrix, 2);
height = size(matrix, 1);
third = size(matrix, 3);
print_std = (third > 1);

if isnumeric(matrix)
  matrix = num2cell(matrix);
  for t = 1:third
    for h=1:height
      for w=1:width
        if(~isempty(format))
          matrix{h, w, t} = num2str(matrix{h, w, t}, format);
        else
          matrix{h, w, t} = num2str(matrix{h, w, t});
        end
      end
    end
  end
end

if(~isempty(textsize))
  fprintf(fid, '\\begin{%s}', textsize);
end

fprintf(fid, '\\begin{tabular}{');

if(~isempty(rowLabels))
  fprintf(fid, 'l');
end
for i=1:width
  fprintf(fid, '%c', alignment);
end
fprintf(fid, '}\r\n');

fprintf(fid, '\\toprule\r\n');

if(~isempty(colLabels))
  if(~isempty(rowLabels))
    fprintf(fid, '&');
  end
  for w=1:width-1
    fprintf(fid, '{%s} & ', colLabels{w});
  end
  fprintf(fid, '{%s}\\\\\\hline\r\n', colLabels{width});
end

for h=1:height
  if(~isempty(rowLabels))
    fprintf(fid, '{%s} & ', rowLabels{h});
  end
  
  
  for w=1:width
    if print_std
      std_str = sprintf('$\\pm$ %s ', matrix{h, w, 2});
    else
      std_str = '';
    end
    this_entry = sprintf('%s %s ', matrix{h, w}, std_str);
    if ~isempty(bold_entries) && bold_entries(h,w)
      this_entry = sprintf('\\textbf{%s}', this_entry);
    end
    if ~isempty(highlight_entries) && highlight_entries(h,w)
      if textcolor
        this_entry = sprintf('\\textcolor{%s}{%s}', ...
          highlight_color, this_entry);
      end
      if textit
        this_entry = sprintf('\\textit{%s}', this_entry);
      end
    end
    if w < width
      this_entry = [this_entry ' & '];
    end
    fprintf(fid, '%s ', this_entry);
  end
  hline = '';
  if ~isempty(hlines) && hlines(h)
    hline = '\hline';
  end
  fprintf(fid, '\\\\ %s \n', hline);
  
end
fprintf(fid, '\\bottomrule\r\n');
fprintf(fid, '\\end{tabular}\r\n');

if(~isempty(textsize))
  fprintf(fid, '\\end{%s}', textsize);
end

fclose(fid);