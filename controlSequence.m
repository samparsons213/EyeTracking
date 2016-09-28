function [control_seq, n_pre_target] = controlSequence(eye_tracking,...
    target_placement)
% Takes a cell array of sequences of eye tracking data, and a cell array of
% sequences of target placements for the data, and returns the sequence of
% target locations for each time stamp of the data after the first target
% has been placed.

% The elements of the original data sequence prior to the target first
% appearing are ignored, so data for each experiment is synchronised across
% people. Each element of control_seq is therefore generally shorter than
% the corresponding element of eye_tracking, and the relation
% control_seq{i} = eye_tracking{i, 2}((n+1):end, :) where n is the number of
% eye tracks recorded before the first target is placed.

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

% control_seq:      n_experiments by 1 cell, each element of which is an n
%                   by 2 series of target coordinates for each time stamp
%                   in the eye-tracking data for that experiment after the
%                   first target has been placed, where n is the number of
%                   time stamps in the eye-tracking data after the first
%                   target has been placed

% n_pre_target:     n_experiments by 1 array of row indices corresponding
%                   to the first row of each experiment in eye_tracking
%                   that occurs after the first target has been placed

% Author:           Sam Parsons
% Date created:     19/09/2016
% Last amended:     27/09/2016

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
%     Main body of code. For eye tracks recorded before target first
%     appears, their target taken to be the following eye track. For eye
%     tracks after target has appeared, their target is taken to be the
%     most recent location of target_placement
%     *********************************************************************

    control_seq = cell(n_experiments, 1);
    n_pre_target = zeros(n_experiments, 1);
    for exp_idx = 1:n_experiments
        n_eye_tracks = length(eye_tracking{exp_idx, 1});
        n_pre_target(exp_idx) = sum(eye_tracking{exp_idx, 1} <...
            target_placement{exp_idx, 1}(1));
        control_seq{exp_idx} =...
            zeros((n_eye_tracks - n_pre_target(exp_idx)), 2);
        for eye_track_row_idx = 1:(n_eye_tracks - n_pre_target(exp_idx))
            target_control_idx =...
                find(target_placement{exp_idx, 1} <=...
                eye_tracking{exp_idx, 1}(eye_track_row_idx +...
                n_pre_target(exp_idx)), 1, 'last');
            try
                control_seq{exp_idx}(eye_track_row_idx, :) =...
                    target_placement{exp_idx, 2}(target_control_idx, :);
            catch ME
                fprintf('experiment: %d, eye_row_idx: %d target_row_idx: %d\n',...
                    exp_idx, eye_track_row_idx, target_control_idx)
                rethrow ME
            end
        end
    end

end