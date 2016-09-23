function [pi_1, P, emission_means, emission_covs] =...
    optimiseParameters(y, p_x, p_x_x_next)
% Returns the parameters that maximise the expected total log-likelihood,
% maximisation step of EM

% Inputs:

% y:                n by 2 by (m+1) array of transformed eye tracking
%                   difference data. First element of third dim is
%                   untransformed, corresponding to 'no movement' latent
%                   state, remaining elements are transformed according to
%                   corresponding columns of l_dirs

% p_x:              n by (m+1) array of probability distributions (each row
%                   sums to 1), giving smoothed distributions
%                   p(x_t | y_1:T, u_1:T)

% p_x_x_next:       (m+1) by (m+1) by (n-1) array of probability
%                   distributions (each matrix sums to 1), giving smoothed
%                   distributions p(x_t, x_t+1 | y_1:T, u_1:T)

% Outputs:

% pi_1:             1 by (m+1) probability distribution (sums to 1) giving
%                   the prior distribution of x_1

% P:                (n+1) by (n+1) matrix of transition probabilties, with
%                   each row summing to 1

% emission_means:   1 by 2 by (m+1) array of mean vectors for the Gaussian 
%                   emission densities for each latent state

% emission_covs:    2 by 2 by (m+1) array of covariance matrices for
%                   the Gaussian emission densities for each latent state

% Author:           Sam Parsons
% Date created:     22/09/2016
% Last amended:     22/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 3
        error('all 3 arguments must be input')
    end
%     y must be a [n 2 m+1] real array
    s_y = size(y);
    if ~(isnumeric(y) && isreal(y) && (ndims(y) == 3) && (s_y(2) == 2))
        error('y must be a [n 2 m+1] real array')
    end
%     p_x must be a [n m+1] array of probability distributions (each row
%     sums to 1)
    if ~(isnumeric(p_x) && isreal(p_x) && all(size(p_x) == [s_y(1), s_y(3)]))
        error('p_x must be a [size(y, 1), size(y, 3)] real array')
    end
    num_tol = 1e-8;
    is_prob_dist = arrayfun(@(row_idx) all(p_x(row_idx, :) >= 0) &&...
        (abs(1 - sum(p_x(row_idx, :))) < num_tol), 1:s_y(1));
    if ~all(is_prob_dist)
        error('all rows of p_x must be probability distributions')
    end
%     p_x_x_next must be a [m+1 m+1 n-1] array of probability matrices
%     (each matrix sums to 1)
    if ~(isnumeric(p_x_x_next) && isreal(p_x_x_next) &&...
            all(size(p_x_x_next) == [s_y(3), s_y(3), s_y(1)-1]))
        err_msg = ['p_x_x_next must be a [size(y, 3), size(y, 3),',...
            ' size(y, 1)-1] real array'];
        error(err_msg)
    end
    is_prob_dist = arrayfun(@(row_idx)...
        all(all(p_x_x_next(:, :, row_idx) >= 0)) &&...
        (abs(1 - sum(sum(p_x_x_next(:, :, row_idx)))) < num_tol),...
        1:(s_y(1)-1));
    if ~all(is_prob_dist)
        error('all matrices of p_x_x_next must be probability distributions')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of function. pi_1 is optimised by setting it to be
%     p_x(1, :), P is optimised by setting each row of it to be
%     proportional to sum(p_x_x_next, 3), with each row summing to 1, each
%     mean vector i in emission_means is optimised by setting it to equal
%     the weighted sum of the corresponding transformation i in y, with
%     each weight proportional to p_x(t, i), and the weights for each i sum
%     to 1. Each covariance matrix i in emission_covs is optimised by
%     setting it to be the weighted sum of 
%     (y(t, :, i) - mu(i))' * (y(t, :, i) - mu(i)), with weights as with
%     emission_means optimisation
%     *********************************************************************

    fprintf('\tOptimising parameters...\n')
    pi_1 = p_x(1, :);
    P = sum(p_x_x_next, 3);
    P = bsxfun(@rdivide, P, sum(P, 2));
    weights = bsxfun(@rdivide, p_x, sum(p_x, 1));
    weights = reshape(weights, s_y(1), 1, s_y(3));
    emission_means = sum(bsxfun(@times, y, weights), 1);
    diffs = num2cell(bsxfun(@minus, y, emission_means), 2);
    diffs_sq = cellfun(@(diff_vec) diff_vec' * diff_vec, diffs,...
        'UniformOutput', false);
    diffs_sq = cell2mat(reshape(diffs_sq, 1, 1, s_y(1), s_y(3)));
    weights = reshape(weights, 1, 1, s_y(1), s_y(3));
    emission_covs = squeeze(sum(bsxfun(@times, diffs_sq, weights), 3));
    fprintf('\tOptimising parameters completed.\n')

end