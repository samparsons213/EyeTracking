function control_seq = controlSequenceHMM(eye_tracking, target_placement)
% Transforms the target placements for 1 person's data into the correct
% form for the proposed HMM - defined to be u(t) = c_s(t) - e_t(t-1), where
% c_s is an element of the control_seq cell obtained from controlSequence.m
% and e_t is the corresponding element of the second column of eye_tracking

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

% Outputs:

% control_seq:      n_experiments by 1 cell, each element of which is an
%                   n-1 by 2 series of target coordinates for differenced
%                   row of eye-tracking data for that experiment, where n
%                   is the number of time stamps in the (pre-differencing)
%                   eye-tracking data

% Author:       Sam Parsons
% Date created: 20/09/2016
% Last amended: 20/09/2016

%     *********************************************************************
%     Check inputs
%     *********************************************************************

%     Both arguments must be input
    if nargin < 2
        error('both eye_tracking and target_placement must be input')
    end
%     Both arguments must be cells
    if ~(iscell(eye_tracking) && iscell(target_placement))
        error('both eye_tracking and target_placement must be cells')
    end
%     Both arguments must have size [n 2] for the same positive integer n
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
%     *********************************************************************

%     *********************************************************************
%     Main body of code. First, the function controlSequence.m is run on
%     the input arguments. Then each the targets are transformed to their
%     differenced equivalents control_seq{exp_idx}(t, :) =
%     control_seq{exp_idx}(t+1, :) - eye_tracking{exp_idx, 2}(t, :)

    control_seq = controlSequence(eye_tracking, target_placement);
    control_seq = cellfun(@(eye_tracks_exp, targets_exp)...
        targets_exp(2:end, :) - eye_tracks_exp(1:(end-1), :),...
        eye_tracking(:, 2), control_seq, 'UniformOutput', false);
end