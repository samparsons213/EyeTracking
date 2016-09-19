function [control_seq] = controlSequence(eye_tracking, target_placement)
% Takes a sequence of eye tracking data, and a sequence of target
% placements for the data, and returns the sequence of target locations for
% each time stamp of the data.

% The elements of the original data sequence prior to the target first
% appearing have their corresponding control sequence elements set to the
% next observed eye tracking location, i.e. u(1) = y(2) where u(t) is the
% control at time t, and y(t) is the eye tracking data at time t.

% Inputs:

% eye_tracking:     1 by 2 cell, where the first element is the series of
%                   time stamps for the eye-tracking data, and the second
%                   element is the eye-tracking data itself (1 by 2 x-y
%                   coordinates for each time stamp)

% target_placement: 1 by 2 cell, where the first element is the series of
%                   time stamps for the target data, and the second element
%                   is the target data itself (1 by 2 x-y coordinates for
%                   each time stamp)

% Outputs:

% control_seq:      n by 2 series of target coordinates for each time stamp
%                   in the eye-tracking data, where n is the number of time
%                   stamps in the eye-tracking data

% Author:           Sam Parsons
% Date created:     19/09/2016
% Last amended:     19/09/2016

    if nargin < 2
        error('both eye_tracking and target_placement must be input')
    end
    if ~(iscell(eye_tracking) && iscell(target_placement))
        error('both eye_tracking and target_placement must be cells')
    end
    s_eye_tracking = size(eye_tracking);
    s_target_placement = size(target_placement);
    if ~((s_eye_tracking(2) == 2) && (s_target_placement(2) == 2) &&...
            (s_eye_tracking(1) == s_target_placement(1)) && (s_eye_tracking(1) > 0))
        err_msg = ['both eye_tracking and target_placement must have size ',...
            '[n 2] for some positive integer n'];
        error(err_msg)
    end
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
    
    control_seq = cell(n_experiments, 1);
    for exp_idx = 1:n_experiments
        n_eye_tracks = length(eye_tracking{exp_idx, 1});
        n_pre_target = sum(eye_tracking{exp_idx, 1} <...
            target_placement{exp_idx, 1}(1));
        control_seq{exp_idx} = [eye_tracking{exp_idx, 2}(2:(n_pre_target+1), :);...
            zeros((n_eye_tracks - n_pre_target), 2)];
        for eye_track_row_idx = (n_pre_target+1):n_eye_tracks
            target_control_idx =...
                find(target_placement{exp_idx, 1} <...
                eye_tracking{exp_idx, 1}(eye_track_row_idx),...
                1, 'last');
            control_seq{exp_idx}(eye_track_row_idx, :) =...
                target_placement{exp_idx, 2}(target_control_idx, :);
        end
    end

end