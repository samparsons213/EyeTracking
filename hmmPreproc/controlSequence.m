function [control_seq, n_pre_target] = controlSequence(ts, eye_tracking, tst, target_placement)
% Takes a sequence of eye tracking data, and a 
% sequence of target placements for the data, and returns the sequence of
% target locations for each time stamp of the data after the first target
% has been placed.

% The elements of the original data sequence prior to the target first
% appearing are ignored, so data for each experiment is synchronised across
% people. control_seq is therefore generally shorter than eye_tracking, and the relation
% control_seq = eye_tracking((n+1):end, :) where n is the number of
% eye tracks recorded before the first target is placed.

% Inputs:

% ts:               timestamps of the eye tracking occurred during the
%                   trial
%
% eye_tracking:     is the eye-tracking data itself for that experiment 
%                   (1 by 2 x-y coordinates for each timestamp)
%
% tst:              timestamps of the targets occurred during the trial
%                  
% target_placement: is the target data itself for that experiment 
%                   (1 by 2 x-y coordinates for each time stamp)

% Outputs:

% control_seq:      n_experiments by 1 cell, each element of which is an n
%                   by 2 series of target coordinates for each time stamp
%                   in the eye-tracking data for that experiment after the
%                   first target has been placed, where n is the number of
%                   time stamps in the eye-tracking data after the first
%                   target has been placed

% n_pre_target:     scalar of the index corresponding
%                   to the first row of each experiment that occurs after 
%                   the first target has been placed

% Author:           Sam Parsons, David Martinez Rego
% Date created:     19/09/2016
% Last amended:     27/09/2016

%     *********************************************************************
%     Check inputs
%     *********************************************************************

%     Both arguments must be input
    if nargin < 4
        error('both eye_tracking and target_placement must be provided')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. For eye tracks recorded before target first
%     appears, eye track is removed. For eye
%     tracks after target has appeared, their target is taken to be the
%     most recent location of target_placement
%     *********************************************************************


    n_eye_tracks = size(eye_tracking, 1);
    n_pre_target = sum(ts < tst(1));

    control_seq = zeros(n_eye_tracks - n_pre_target, 2);
    nextAdd = 1;
    currentTarget = 1;
    for eye_track_row = (n_pre_target+1):n_eye_tracks,
        % Jump to the next one if needed
        if((currentTarget < length(tst)) && (ts(eye_track_row) >= tst(currentTarget+1))),
            currentTarget = currentTarget+1;
        end
        % Add current target
        if(ts(eye_track_row) >= tst(currentTarget)),
            control_seq(nextAdd, :) = target_placement(currentTarget, :);
            nextAdd = nextAdd+1;
        end
    end