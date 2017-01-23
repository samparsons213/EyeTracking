function [y, zero_prob, u] = hmmPreproc(ts, eye_tracking, tst, target_placement, axis_dir, axis_orth_dir, diff_length)
% Produces the transformations of the difference vectors of eye_tracking
% data, and corresponding control sequence for each experiment on one
% person's data

% Inputs:
%
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

% axis_dir:         2 by m array of canonical latent directions, where each
%                   column is a unit vector

% axis_orth_dir:    2 by m array of vectors orthogonal to each of the
%                   canonical latent directions, where each column is a
%                   unit vector

% diff_length:  positive integer giving the gaps between eye tracks for
%               computing difference vectors, e.g. if diff_length = 3 and
%               T = 7 for some element i of y, then the difference vectors
%               (prior to any transformations) would be
%               y{i}(4, :) - y{i}(1, :); y{i}(7, :)-y{i}(4, :). If not
%               input then its default value is 1

% Outputs:

% y:                a T by 2 by (m+1) array of transformed difference
%                   vectors, where T is the number of difference vectors
%                   (IMPORTANT) after the first target has appeared

% zero_prob:        a T by n binary array, where zeros correspond to zero
%                   posterior probability for that direction at that time,
%                   and ones correspond to positive posterior probability
%                   for that direction at that time, with T as in y


% u:                a T by 2 series of target coordinates for differenced
%                   row of eye-tracking data for that experiment, with T as
%                   in y

% Author:       Sam Parsons, David Martinez Rego
% Date created: 27/09/2016
% Last amended: 12/10/2016

%     *********************************************************************
%     Check inputs
%     *********************************************************************


    % Utilities
    isNumMatrix = @(x) isnumeric(x) && ismatrix(x);
    
    if nargin < 4
        error('the first 4 arguments must be provided')
    end
    
    if ~(isNumMatrix(eye_tracking) && isNumMatrix(target_placement)),
        error('both eye_tracking and target_placement must be numerical matrices')
    end
    
    if ~iscolumn(ts)
        error('ts must be a column vector')
    end
    
    if ~iscolumn(tst)
        error('tst must be a column vector')
    end
    
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Here are the steps done:
%     1. we check to make sure we have targets in this experiment, otherwise we exit.
%     2, axisProjection is run, producing the transformed difference
%     data needed (including rows prior to the first target appearing in
%     each experiment). 
%     3, controlSequence.m is run , producing the target coordinates at each 
%     eye tracking time stamp after the first target has been placed. 
%     4. the rows of y and zero_prob prior to the first target appearing are removed
%     *********************************************************************

    if isempty(target_placement),
        y = [];
        zero_prob = [];
        u = [];
        return;
    end
    
   
    [y, zero_prob] = axisProjection(eye_tracking, axis_dir, axis_orth_dir, diff_length);
    [control_seq, n_pre_target] = controlSequence(ts, eye_tracking, tst, target_placement);
    
    
    % Calculat the difference between the target and the previous gaze
    % position
    u = control_seq(diff_length:diff_length:end, :) -...
        eye_tracking((n_pre_target+1):diff_length:(end-diff_length), :);
    
    u_length = size(u, 1);
    
    % Remove eye movement prior to the first target occurrying 
    y = y((end - u_length + 1):end, :, :);
    zero_prob = zero_prob((end - u_length + 1):end, :);
end