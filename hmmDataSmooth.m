function [y, zero_prob, u] = hmmDataSmooth(eye_tracking, target_placement,...
    l_dirs, orth_l_dirs, ma_length)
% Produces the transformations of the difference vectors of eye_tracking
% data, and corresponding control sequence for each experiment on one
% person's data. Instead of lagged differences, moving averages are used to
% smooth out noise from random eye movements.

% Inputs:

% eye_tracking:     n_experiments by 2 cell, where each element in the
%                   first column is a series of time stamps for one
%                   experiment of eye-tracking data, and each element in
%                   the second column is the eye-tracking data itself for
%                   that experiment (1 by 2 x-y coordinates for each time
%                   stamp)

% target_placement: n_experiments by 2 cell, where each element in the
%                   first column is a series of time stamps for the targets
%                   in one experiment of eye-tracking data, and each
%                   element in the second column is the target data
%                   itself for that experiment (1 by 2 x-y coordinates for
%                   each time stamp)

% l_dirs:           2 by m array of canonical latent directions, where each
%                   column is a unit vector

% orth_l_dirs:      2 by m array of vectors orthogonal to each of the
%                   canonical latent directions, where each column is a
%                   unit vector

% ma_length:        positive integer giving the length of moving window for
%                   moving average of eye tracks, e.g. if ma_length = 3
%                   then the difference vectors comprising each
%                   y{i}(:, :, 1) would be differences of the smoothed
%                   sequence s{i}(t)=mean(eye_tracking{i, 2}(t:t+ma-1, :)).
%                   If not input then its default value is 1

% Outputs:

% y:                n_experiments by 1 cell, where each element is a
%                   T by 2 by (m+1) array of transformed difference
%                   vectors, where T is the number of difference vectors
%                   after the first target has appeared

% zero_prob:        n_experiments by 1 cell, where each element is a
%                   T by n binary array, where zeros correspond to zero
%                   posterior probability for that direction at that time,
%                   and ones correspond to positive posterior probability
%                   for that direction at that time, with T as in y


% u:                n_experiments by 1 cell, each element of which is an
%                   T by 2 series of target coordinates for differenced
%                   row of eye-tracking data for that experiment, with T as
%                   in y

% Author:       Sam Parsons
% Date created: 01/11/2016
% Last amended: 01/11/2016

%     *********************************************************************
%     Check inputs
%     *********************************************************************

%     The first 4 arguments must be input
    if nargin < 4
        error('the first 4 arguments must be input')
    end
%     Both eye_tracking and target_placement must be cells
    if ~(iscell(eye_tracking) && iscell(target_placement))
        error('both eye_tracking and target_placement must be cells')
    end
%     Both eye_tracking and target_placement must have size [n 2] for the
%     same positive integer n
    s_eye_tracking = size(eye_tracking);
    s_target_placement = size(target_placement);
    if ~(ismatrix(eye_tracking) && ismatrix(target_placement) &&...
            (s_eye_tracking(2) == 2) && (s_target_placement(2) == 2) &&...
            (s_eye_tracking(1) == s_target_placement(1)) && (s_eye_tracking(1) > 0))
        err_msg = ['both eye_tracking and target_placement must have size ',...
            '[n 2] for the same positive integer n'];
        error(err_msg)
    end
%     eye_tracking{exp_idx, 1} must be a column vector for all experiments
%     exp_idx, eye_tracking{exp_idx, 2} must be a matrix with same number
%     of rows as eye_tracking{exp_idx, 1} and 2 columns. Similar for
%     target_placement
    n_experiments = s_eye_tracking(1);
    for exp_idx = 1:n_experiments
        if ~iscolumn(eye_tracking{exp_idx, 1})
            err_msg = ['eye_tracking{', num2str(exp_idx),...
                ',1} must be a column vector'];
            error(err_msg)
        end
        if ~all(size(eye_tracking{exp_idx, 2}) ==...
                [length(eye_tracking{exp_idx, 1}) 2])
            err_msg = ['eye_tracking{', num2str(exp_idx), '2} must have ',...
                'size [n 2], where n is the number of rows of ',...
                'eye_tracking{', num2str(exp_idx), '1}'];
            error(err_msg)
        end
        if ~iscolumn(target_placement{exp_idx, 1})
            err_msg = ['target_placement{', num2str(exp_idx),...
                ',1} must be a column vector'];
            error(err_msg)
        end
        if ~all(size(target_placement{exp_idx, 2}) ==...
                [length(target_placement{exp_idx, 1}) 2])
            err_msg = ['target_placement{', num2str(exp_idx), '2} must have ',...
                'size [n 2], where n is the number of rows of ',...
                'target_placement{', num2str(exp_idx), '1}'];
            error(err_msg)
        end
    end
%     l_dirs must be a [2 m] numeric array for some m
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
%     orth_l_dirs must be a [2 m] numeric array for same m as l_dirs
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
        smallest_n = min(arrayfun(@(exp_idx) size(eye_tracking{exp_idx, 1}, 1),...
            1:n_experiments));
        ma_length = max(1, ma_length);
        ma_length = min((smallest_n-1), ma_length);
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. First, all experiments are checked to make sure
%     they have targets in them, and any without targets are discarded.
%     Next, yTransformHMM.m is run, producing the transformed difference
%     data needed (including rows prior to the first target appearing in
%     each experiment). Second, controlSequence.m is run , producing the
%     target coordinates at each eye tracking time stamp after the first
%     target has been placed. Lastly, the rows of y and zero_prob prior to
%     the first target appearing are removed
%     *********************************************************************

    any_targets = arrayfun(@(exp_idx)...
        size(target_placement{exp_idx, 1}, 1) > 0,...
        1:size(target_placement, 1));
    eye_tracking = eye_tracking(any_targets, :);
    target_placement = target_placement(any_targets, :);
    [y, zero_prob] = yTransformHMMSmooth(eye_tracking(:, 2), l_dirs, orth_l_dirs,...
        ma_length);
    [control_seq, n_pre_target] = controlSequence(eye_tracking,...
        target_placement);
    n_experiments = length(control_seq);
    u = arrayfun(@(exp_idx) control_seq{exp_idx}((1+ma_length):ma_length:end, :) -...
        eye_tracking{exp_idx, 2}((n_pre_target(exp_idx)+1):ma_length:(end-ma_length), :),...
        (1:n_experiments)', 'UniformOutput', false);
% %     y_start = arrayfun(@(exp_idx)...
% %         ceil((n_pre_target(exp_idx)+1) / diff_length), 1:n_experiments);
% %     y = arrayfun(@(exp_idx) y{exp_idx}(y_start(exp_idx):end, :, :),...
% %         (1:n_experiments)', 'UniformOutput', false);
% %     zero_prob = arrayfun(@(exp_idx)...
% %         zero_prob{exp_idx}(y_start(exp_idx):end, :),...
% %         (1:n_experiments)', 'UniformOutput', false);
    u_length = arrayfun(@(exp_idx) size(u{exp_idx}, 1), (1:length(u))');
    y = arrayfun(@(exp_idx) y{exp_idx}((end - u_length(exp_idx) + 1):end, :, :),...
        (1:n_experiments)', 'UniformOutput', false);
    zero_prob = arrayfun(@(exp_idx)...
        zero_prob{exp_idx}((end - u_length(exp_idx) + 1):end, :),...
        (1:n_experiments)', 'UniformOutput', false);

end