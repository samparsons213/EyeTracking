function p_ugx = condProbUX_ManyPeople(control_seq_hmm, l_dirs, epsilon)
% Returns the posterior densities of each element in the control sequence
% given each canonical latent direction for each (differenced) time step in
% each experiment for multiple people's data

% Inputs:

% control_seq_hmm:  n_people by 1 cell, each element of which is an
%                   n_experiments by 1 cell, each element of which is an n
%                   by 2 array of control directions for the differenced
%                   data y for some positive n (not same n across
%                   experiments or people)

% l_dirs:           2 by m array of canonical latent directions, where each
%                   column is a unit vector

% epsilon:          real number in (0, 0.5) giving the closest acceptable
%                   norm-difference (see below) to 0 and 1

% Outputs:

% p_ugx:            n_people by 1 cell, each element of wihch is an 
%                   n_experiments by 1 cell, each element of which is an n
%                   by m+1 array of conditional densities
%                   (first column relates to 'no movement' latent state,
%                   last m columns relate to corresponding columns in
%                   l_dirs) (n not same across experiments, equal to n in
%                   control_seq_hmm for each experiment). Note that each
%                   row in each cell does ***not*** represent one
%                   unnormalised distribution, but each element of the row
%                   is from a different conditional density

% Author:       Sam Parsons
% Date created: 27/09/2016
% Last amended: 27/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
%     All  arguments must be input
    if nargin < 3
        error('all 3 arguments must be input')
    end
%     control_seq_hmm must be an [n_people 1] cell array
    if ~(iscell(control_seq_hmm) && iscolumn(control_seq_hmm))
        error('control_seq_hmm must be a [n_people 1] cell array')
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
%     epsilon must be a real scalar in (0, 0.5)
    if ~(isscalar(epsilon) && isnumeric(epsilon) && isreal(epsilon) &&...
            (epsilon > 0) && (epsilon < 0.5))
        error('epsilon must be a real scalar in (0, 0.5)')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Runs condProbUXAll.m for each person's
%     control_seq_hmm data (i.e. each element of control_seq_hmm)
%     *********************************************************************

    p_ugx = cellfun(@(control_seq_hmm_person)...
        condProbUXAll(control_seq_hmm_person, l_dirs, epsilon),...
        control_seq_hmm, 'UniformOutput', false);

end