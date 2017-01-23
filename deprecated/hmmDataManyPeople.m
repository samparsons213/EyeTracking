function [y, zero_prob, u] = hmmDataManyPeople(eye_tracking,...
    target_placement, l_dirs, orth_l_dirs, diff_length)
% Produces the transformations of the difference vectors of eye_tracking
% data, and corresponding control sequence for each experiment on mutiple
% people's data

% Inputs:

% eye_tracking:     n_people by 1 cell, each element of which is an
%                   n_experiments by 2 cell (n_experiments is not
%                   necessarily constant across people), where each element
%                   in the first column is a series of time stamps for one
%                   experiment of eye-tracking data, and each element in
%                   the second column is the eye-tracking data itself for
%                   that experiment (1 by 2 x-y coordinates for each time
%                   stamp)

% target_placement: n_people by 1 cell, each element of which is an
%                   n_experiments by 2 cell (n_experiments is not
%                   necessarily constant across people), where each element
%                   in the first column is a series of time stamps for the
%                   targets in one experiment of eye-tracking data, and
%                   each element in the second column is the target data
%                   itself for that experiment (1 by 2 x-y coordinates for
%                   each time stamp)

% l_dirs:           2 by m array of canonical latent directions, where each
%                   column is a unit vector

% orth_l_dirs:      2 by m array of vectors orthogonal to each of the
%                   canonical latent directions, where each column is a
%                   unit vector

% diff_length:  positive integer giving the gaps between eye tracks for
%               computing difference vectors, e.g. if diff_length = 3 and
%               T = 7 for some element i of y, then the difference vectors
%               (prior to any transformations) would be
%               y{i}(4, :) - y{i}(1, :); y{i}(7, :)-y{i}(4, :). If not
%               input then its default value is 1

% Outputs:

% y:                n_people by 1 cell, each element of which is an
%                   n_experiments by 1 cell (n_experiments is not
%                   necessarily constant across people), where each element
%                   is a T by 2 by (m+1) array (T not necessarily constant
%                   across people or experiments) of transformed difference
%                   vectors, where T is the number of difference vectors
%                   after the first target has appeared

% zero_prob:        n_people by 1 cell, each element of which is an
%                   n_experiments by 1 cell (n_experiments is not
%                   necessarily constant across people), where each element
%                   is a T by n binary array,  zeros correspond to
%                   zero posterior probability for wherethat direction at that
%                   time, and ones correspond to positive posterior
%                   probability for that direction at that time, with T as
%                   in y

% u:                n_people by 1 cell, each element of which is an
%                   n_experiments by 1 cell (n_experiments is not
%                   necessarily constant across people), each element of
%                   which is a T by 2 series of target coordinates for
%                   differenced row of eye-tracking data for that
%                   experiment, with T as in y

% Author:       Sam Parsons
% Date created: 27/09/2016
% Last amended: 27/09/2016

%     *********************************************************************
%     Check inputs
%     *********************************************************************

%     The first 4 arguments must be input
    if nargin < 4
        error('first 4 arguments must be input')
    end
%     Both eye_tracking and target_placement must be cells
    if ~(iscell(eye_tracking) && iscell(target_placement))
        error('both eye_tracking and target_placement must be cells')
    end
%     Both eye_tracking and target_placement must have size [n 1] for the
%     same positive integer n
    l_eye_tracking = length(eye_tracking);
    l_target_placement = length(target_placement);
    if ~(iscolumn(eye_tracking) && iscolumn(target_placement) &&...
            (l_eye_tracking == l_target_placement) && (l_eye_tracking > 0))
        err_msg = ['both eye_tracking and target_placement must have size ',...
            '[n 1] for the same positive integer n'];
        error(err_msg)
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
%     If input, diff_length must be a positive integer no greater than x-1,
%     where x is the smallest n across experiments where
%     size(y{exp_idx}) = [n 2]. If outside the acceptable range, it is
%     placed inside at the nearest boundary. If not input, it is set to its
%     default value of 1
    if nargin == 3
        diff_length = 1;
    else
        if ~(isnumeric(diff_length) && isreal(diff_length) &&...
                isscalar(diff_length) &&...
                (diff_length - round(diff_length) < num_tol))
            error('diff_length must be a scalar integer')
        end
        n_people = length(eye_tracking);
        smallest_n = zeros(n_people, 1);
        for person_idx = 1:length(eye_tracking)
            n_experiments = size(eye_tracking{person_idx}, 1);
            smallest_n(person_idx) = min(arrayfun(@(exp_idx)...
                size(eye_tracking{person_idx}{exp_idx, 1}, 1), 1:n_experiments));
        end
        smallest_n = min(smallest_n);
        diff_length = max(1, diff_length);
        diff_length = min((smallest_n-1), diff_length);
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. hmmData.m is run for each element of eye_tracking
%     and target_placement
%     *********************************************************************

    [y, zero_prob, u] = cellfun(@(eye_track_person, target_placing_person)...
        hmmData(eye_track_person, target_placing_person, l_dirs,...
        orth_l_dirs, diff_length), eye_tracking, target_placement,...
        'UniformOutput', false);
end