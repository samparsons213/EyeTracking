function p_ugx = condProbUXAll(control_seq_hmm, l_dirs, median_norm_u, epsilon)
% Returns the posterior densities of each element in the control sequence
% given each canonical latent direction for each (differenced) time step in
% each experiment for one person's data

% Inputs:

% control_seq_hmm:  n_experiments by 1 cell, each element of which is an n
%                   by 2 array of control directions for the differenced
%                   data y for some positive n (not same n across
%                   experiments)

% l_dirs:           2 by m array of canonical latent directions, where each
%                   column is a unit vector

% median_norm_u:    positive real number giving the median norm of the
%                   control directions in the experiment

% epsilon:          real number in (0, 0.5) giving the closest acceptable
%                   norm-difference (see below) to 0 and 1

% Outputs:

% p_ugx:            n_experiments by 1 cell, each element of which is an n
%                   by m+1 array of conditional densities
%                   (first column relates to 'no movement' latent state,
%                   last m columns relate to corresponding columns in
%                   l_dirs) (n not same across experiments, equal to n in
%                   control_seq_hmm for each experiment). Note that each
%                   row in each cell does ***not*** represent one
%                   unnormalised distribution, but each element of the row
%                   is from a different conditional density

% Author:       Sam Parsons
% Date created: 21/09/2016
% Last amended: 17/11/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 4
        error('all 4 arguments must be input')
    end
%     control_seq_hmm must conform to the output of controlSequenceHMM.m
    if ~(iscell(control_seq_hmm) && iscolumn(control_seq_hmm))
        error('control_seq_hmm must be a column vector cell array')
    end
    n_experiments = length(control_seq_hmm);
    correct = arrayfun(@(exp_idx) isnumeric(control_seq_hmm{exp_idx}) &&...
        ismatrix(control_seq_hmm{exp_idx}) &&...
        isreal(control_seq_hmm{exp_idx}) &&...
        (size(control_seq_hmm{exp_idx}, 1) > 0) &&...
        (size(control_seq_hmm{exp_idx}, 2) == 2), 1:n_experiments);
    if ~all(correct)
        err_msg = ['each element of control_seq_hmm must be a real matrix',...
            ' with 2 columns and at least 1 row'];
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
%     median_norm_u must be a positive real scalar
    if ~(isscalar(median_norm_u) && isnumeric(median_norm_u) &&...
            isreal(median_norm_u) && (median_norm_u > 0))
        error('median_norm_u must be a positive real scalar')
    end
%     epsilon must be a real scalar in (0, 0.5)
    if ~(isscalar(epsilon) && isnumeric(epsilon) && isreal(epsilon) &&...
            (epsilon > 0) && (epsilon < 0.5))
        error('epsilon must be a real scalar in (0, 0.5)')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. For each experiment, the norm of the largest
%     control direction is found. Then, the function condProbUX.m is run on
%     each row of control_seq_hmm{exp_idx} in turn with median_norm_u
%     constant inside each experiment. The normaliser of the conditional
%     densities p(u | x) for x = 2,...,(dim_x+1) (i.e. movement states) is
%     calculated here as 
%     integral(@(theta) 2\log(2) - log(1-cos(theta)), 0, pi) = 3.2664
%     (4 d.p.)
%     *********************************************************************

    movement_z = integral(@(theta) 2\log(2) - log(1-cos(theta)), 0, pi);
    p_ugx = cell(n_experiments, 1);
    for exp_idx = 1:n_experiments
        n = size(control_seq_hmm{exp_idx}, 1);
        p_ugx{exp_idx} = cell2mat(arrayfun(@(row_idx)...
            condProbUX(control_seq_hmm{exp_idx}(row_idx, :)', l_dirs,...
            median_norm_u, movement_z, epsilon), (1:n)', 'UniformOutput',...
            false));
    end

end