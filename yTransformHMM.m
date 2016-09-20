function [y, zero_prob] = yTransformHMM(y, l_dirs, orth_l_dirs)
% Transforms the T by 2 eye tracking data for each experiment for one
% person into the form needed for inference in the proposed HMM.

% Inputs:

% y:            n_experiments by 1 cell, where each element is a T by 2
%               array of eye tracking locations

% l_dirs:       2 by n array of canonical latent directions, where each
%               column is a unit vector

% orth_l_dirs:  2 by n array of vectors orthogonal to each of the canonical
%               latent directions, where each column is a unit vector

% Outputs:

% y:            n_experiments by 1 cell, where each element is a (T-1) by 2
%               by (n+1) array of transformed difference vectors

% zero_prob:    n_experiments by 1 cell, where each element is a (T-1) by n
%               binary array, where zeros correspond to zero posterior
%               probability for that direction at that time, and ones
%               correspond to positive posterior probability for that
%               direction at that time

% Author:       Sam Parsons
% Date created: 20/09/2016
% Last amended: 20/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 3
        error('all 3 arguments must be input')
    end
%     y must be a cell
    if ~iscell(y)
        error('y must be a cell')
    end
%     y must be [n 1] for some n
    if ~iscolumn(y)
        error('y must be an n by 1 cell for some positive integer n')
    end
%     All elements of y must be [n 2] numeric arrays for some n
    n_experiments = length(y);
    correct = arrayfun(@(exp_idx) isnumeric(y{exp_idx}) &&...
        ismatrix(y{exp_idx}) && (size(y{exp_idx}, 2) == 2),...
        1:n_experiments);
    if ~all(correct)
        err_msg = ['data for experiments ', num2str(find(~correct)),...
            ' are either not numeric, or not of size [n 2] for some',...
            ' positive integer n'];
        error(err_msg)
    end
%     l_dirs must be a [2 n] numeric array for some n
    if ~(isnumeric(l_dirs) && ismatrix(l_dirs) && (size(l_dirs, 1) == 2))
        err_msg = ['l_dirs must be a 2 by n numeric array for some',...
            ' positive integer n'];
        error(err_msg)
    end
%     All columns of l_dirs must be unit vectors
    n_l_dirs = size(l_dirs, 2);
    num_tol = 1e-8;
    correct = arrayfun(@(dir_idx)...
        abs(1 - (l_dirs(:, dir_idx)' * l_dirs(:, dir_idx))) < num_tol,...
        1:n_l_dirs);
    if ~all(correct)
        error('all columns in l_dirs must be unit vectors')
    end
%     orth_l_dirs must be a [2 n] numeric array for same n as l_dirs
    if ~(isnumeric(orth_l_dirs) && ismatrix(orth_l_dirs) &&...
            all(size(orth_l_dirs) == [2 n_l_dirs]))
        err_msg = ['orth_l_dirs must be a numeric array of the same size',...
            ' as l_dirs'];
        error(err_msg)
    end
%     All columns of orth_l_dirs must be unit vectors
    correct = arrayfun(@(dir_idx)...
        abs(1 - (orth_l_dirs(:, dir_idx)' * orth_l_dirs(:, dir_idx))) < num_tol,...
        1:n_l_dirs);
    if ~all(correct)
        error('all columns in orth_l_dirs must be unit vectors')
    end
%     All columns of orth_l_dirs must be orthogonal to corresponding column
%     of l_dirs
    correct = arrayfun(@(dir_idx)...
        abs(l_dirs(:, dir_idx)' * orth_l_dirs(:, dir_idx)) < num_tol,...
        1:n_l_dirs);
    if ~all(correct)
        err_msg = ['all columns of orth_l_dirs must be orthogonal to the',...
            ' corresponding column of l_dirs'];
        error(err_msg)
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Perform differencing on y, then the transforming
%     required for each latent direction. Record which latent directions
%     will have zero posterior probability (<=> non-positive movement in
%     that direction)
%     *********************************************************************

    zero_prob = cell(size(y));
    for exp_idx = 1:n_experiments
        y_diff = diff(y{exp_idx});
        t = size(y_diff, 1);
        y_transformed = zeros(t, 2*n_l_dirs);
        x1 = y_diff * l_dirs;
        zero_prob{exp_idx} = x1 > 0;
        x1(zero_prob{exp_idx}) = log(x1(zero_prob{exp_idx}));
        y_transformed(:, 1:2:(2*n_l_dirs)) = x1;
        y_transformed(:, 2:2:(2*n_l_dirs)) = y_diff * orth_l_dirs;
        y{exp_idx} = cat(3, y_diff, reshape(y_transformed, t, 2, n_l_dirs));
    end

end