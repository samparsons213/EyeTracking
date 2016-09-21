function p_ugx = condProbUX(u, l_dirs, median_norm_u, epsilon)
% Returns unnormalised conditional probabilities of the control direction
% u given each of the latent directions and the 'no movement' latent state

% Inputs:

% u:            2 by 1 control direction

% l_dirs:       2 by n array of latent directions, where each column is a
%               unit vector

% median_norm_u:positive real number giving the median norm of the control
%               directions in the experiment

% epsilon:      real number in (0, 0.5) giving the closest acceptable
%               norm-difference (see below) to 0 and 1

% Outputs:

% p_ugx:        1 by (n+1) probability distribution, with the first n
%               elements giving the unnormalised conditional probability of
%               the corresponding column of l_dirs given the control
%               direction u, and the last element giving the unnormalised
%               probability for the 'no movement' latent state. Note that
%               this is not itself an unnormalised distribution, rather
%               each element comes from an unnormalised distribution

% Author:       Sam Parsons
% Date created: 20/09/2016
% Last amended: 21/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 3
        error('all 3 arguments must be input')
    end
%     u must be a numeric vector of size [2 1]
    if ~isnumeric(u)
        error('u must be numeric')
    end
    if ~(iscolumn(u) && (length(u) == 2))
        error('u must be a column vector of size [2 1]')
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
    norm_u = norm(u);
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
%     Main body of code. Produces a positive score for each of the latent
%     directions, and for the 'no movement' latent state. These scores will
%     then be used to weight the transition matrix before it is
%     renormalised (row by row). For the latent directions, we take the
%     norm of the difference between (normalised) u and each column of
%     l_dirs, and half them so they're all <= 1 and their logs are all
%     non-positive. These are then thresholded to be at least epsilon and
%     at most 1-epsilon, to avoid log(0) and log(1). Thresholded values are
%     logged, and then normalised (minus sign cancels after normalisation,
%     so it is not not bothered with). For the 'no movement' latent state,
%     the norm of u as a fraction of max_norm_u is negatively logged (after
%     thresholding as before)
%     *********************************************************************

    norm_u_ratio = norm_u / median_norm_u;
    u = u ./ norm(u);
    diffs = 0.5 .* arrayfun(@(dir_idx) norm(u - l_dirs(:, dir_idx)),...
        1:size(l_dirs, 2));
    exp_scores = [diffs, norm_u_ratio];
    exp_scores = max(epsilon, exp_scores);
    exp_scores = min(1-epsilon, exp_scores);
    p_ugx = -log(exp_scores);

end