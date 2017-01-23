function [y, zero_prob] = yTransformHMMSmooth(y, l_dirs, orth_l_dirs,...
    ma_length)
% Transforms the T by 2 eye tracking data for each experiment for one
% person into the form needed for inference in the proposed HMM. Difference
% vectors of a specified moving average window are computed from
% eye_tracking data, and projected onto the orthogonal basis implied by
% each canonical direction in l_dirs. When the projection onto the primary
% basis vector (i.e. the canonical direction itself) for each basis is
% non-positive, the corresponding element of the logical array zero_prob is
% set to false to indicate that its corresponding emission density is 0.

% Inputs:

% y:            n_experiments by 1 cell, where each element is a T by 2
%               array of eye tracking locations

% l_dirs:       2 by n array of canonical latent directions, where each
%               column is a unit vector

% orth_l_dirs:  2 by n array of vectors orthogonal to each of the canonical
%               latent directions, where each column is a unit vector

% ma_length:    positive integer giving the length of moving window for
%               moving average of eye tracks, e.g. if ma_length = 3
%               then the difference vectors comprising each
%               y{i}(:, :, 1) would be differences of the smoothed
%               sequence s{i}(t)=mean(eye_tracking{i, 2}(t:t+ma-1, :)).
%               If not input then its default value is 1

% Outputs:

% y:            n_experiments by 1 cell, where each element is a (x-1) by 2
%               by (n+1) array of transformed difference vectors, where
%               x=length(1:diff_length:T)=1+floor((T-1)/diff_length)

% zero_prob:    n_experiments by 1 cell, where each element is a (x-1) by n
%               binary array, where zeros correspond to zero posterior
%               probability for that direction at that time, and ones
%               correspond to positive posterior probability for that
%               direction at that time, and x is as in y above

% Author:       Sam Parsons
% Date created: 01/11/2016
% Last amended: 01/11/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     First 3 arguments must be input
    if nargin < 3
        error('the first 3 arguments must be input')
    end
%     y must be a cell
    if ~iscell(y)
        error('y must be a cell')
    end
%     y must be [n 1] for some
    if ~iscolumn(y)
        error('y must be an n by 1 cell for some positive integer n')
    end
%     All elements of y must be [n 2] numeric arrays for some n > 1
    n_experiments = length(y);
    correct = arrayfun(@(exp_idx) isnumeric(y{exp_idx}) &&...
        ismatrix(y{exp_idx}) && (size(y{exp_idx}, 2) == 2) &&...
        (size(y{exp_idx}, 1) > 1), 1:n_experiments);
    if ~all(correct)
        err_msg = ['data for experiments ', num2str(find(~correct)),...
            ' are either not numeric, or not of size [n 2] for some',...
            ' positive integer n > 1'];
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
%     If input, ma_length must be a positive integer no greater than x-1,
%     where x is the smallest n across experiments where
%     size(y{exp_idx}) = [n 2]. If outside the acceptable range, it is
%     placed inside at the nearest boundary. If not input, it is set to its
%     default value of 1
    if nargin == 3
        ma_length = 1;
    else
        if ~(isnumeric(ma_length) && isreal(ma_length) &&...
                isscalar(ma_length) &&...
                (ma_length - round(ma_length) < num_tol))
            error('diff_length must be a scalar integer')
        end
        smallest_n = min(arrayfun(@(exp_idx) size(y{exp_idx}, 1),...
            1:n_experiments));
        ma_length = max(1, ma_length);
        ma_length = min((smallest_n-1), ma_length);
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Perform moving average smoothing on y, then the
%     transforming required for each latent direction. Record which latent
%     directions will have zero posterior probability (<=> non-positive
%     movement in that direction)
%     *********************************************************************

    zero_prob = cell(size(y));
    for exp_idx = 1:n_experiments
        t = size(y{exp_idx}, 1);
        y_smoothed = mean(cell2mat(arrayfun(@(w_idx)...
            y{exp_idx}((1:(t-(ma_length-1)))+w_idx, :), 0:(ma_length-1),...
            'UniformOutput', false)), 2);
        y_diff = diff(y_smoothed);
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