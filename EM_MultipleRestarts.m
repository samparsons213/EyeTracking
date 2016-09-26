function [pi_1, P, emission_means, emission_covs] = EM_MultipleRestarts(y,...
    p_ugx, zero_probs, n_iters, n_restarts)
% Performs n_iters iterations of EM on one person's data for one experiment

% Inputs:

% y:                n by 2 by (m+1) array of transformed eye tracking
%                   difference data. First element of third dim is
%                   untransformed, corresponding to 'no movement' latent
%                   state, remaining elements are transformed according to
%                   corresponding columns of l_dirs

% p_ugx:            n by (m+1) array of unnormalised conditional
%                   probabilities of u_t | x_t for each possible latent
%                   state - first element corresponds to 'no movement'
%                   latent state, last n elements correspond to each
%                   direction in l_dirs

% zero_probs:       n by m binary array, where zeros correspond to the
%                   latent directions that have zero posterior probability,
%                   and ones to all other latent directions

% n_iters:          positive integer giving number of iterations of EM to
%                   run

% n_restarts:       positive integer giving number of restarts of EM
%                   procedure

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
%     All  arguments must be input
    if nargin < 5
        error('all 5 arguments must be input')
    end
%     y must be a [n 2 m+1] real array
    s_y = size(y);
    if ~(isnumeric(y) && isreal(y) && (ndims(y) == 3) && (s_y(2) == 2))
        error('y must be a [n 2 m+1] real array')
    end
%     p_ugx must be a [n m+1] array of positive reals
    if ~(isnumeric(p_ugx) && isreal(p_ugx) && ismatrix(p_ugx) &&...
            all(size(p_ugx) == [s_y(1), s_y(3)]) && all(p_ugx(:) > 0))
        err_msg = ['p_ugx must be a [size(y, 1) size(y, 3)] array of',...
            ' positive reals'];
        error(err_msg)
    end
%     zero_probs must be [n m+1] binary array
    if ~(isnumeric(zero_probs) && isreal(zero_probs) && ismatrix(zero_probs) &&...
            all(size(zero_probs) == [s_y(1), s_y(3) - 1]) &&...
            all((zero_probs(:) == 0) | (zero_probs(:) == 1)))
        error('zero_probs must be a [size(y, 1) size(y, 3)-1] binary array')
    end
%     n_iters must be a positive integer
    if ~(isnumeric(n_iters) && isreal(n_iters) && isscalar(n_iters) &&...
            (n_iters == round(n_iters)) && (n_iters > 0))
        error('n_iters must be a positive integer')
    end
%     n_restarts must be a positive integer
    if ~(isnumeric(n_restarts) && isreal(n_restarts) && isscalar(n_restarts) &&...
            (n_restarts == round(n_restarts)) && (n_restarts > 0))
        error('n_restarts must be a positive integer')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Runs EM.m n_restarts, and selects the most likely
%     parameters from those found at each restart.
%     *********************************************************************

    likelihoods = zeros(1, n_restarts);
    pi_1 = zeros(n_restarts, s_y(3));
    P = zeros(s_y(3), s_y(3), n_restarts);
    emission_means = zeros(n_restarts, 2, s_y(3));
    emission_covs = zeros(2, 2, s_y(3), n_restarts);
    for restart_idx = 1:n_restarts
        fprintf('\n\nStarting parameter estimation procedure %d...\n\n',...
            restart_idx)
        [pi_1(restart_idx, :), P(:, :, restart_idx),...
            emission_means(restart_idx, :, :),...
            emission_covs(:, :, :, restart_idx)] =...
            generateRandomParameters(s_y(3));
        [pi_1(restart_idx, :), P(:, :, restart_idx),...
            emission_means(restart_idx, :, :),...
            emission_covs(:, :, :, restart_idx)] = EM(y, p_ugx,...
            pi_1(restart_idx, :), P(:, :, restart_idx),...
            emission_means(restart_idx, :, :),...
            emission_covs(:, :, :, restart_idx), zero_probs, n_iters);
        p_x = forwardBackward(y, p_ugx, pi_1(restart_idx, :),...
            P(:, :, restart_idx), emission_means(restart_idx, :, :),...
            emission_covs(:, :, :, restart_idx), zero_probs);
        conditional_likelhoods = cell2mat(arrayfun(@(x_idx)...
            mvnpdf(y(:, :, x_idx), emission_means(restart_idx, :, x_idx),...
            emission_covs(:, :, x_idx, restart_idx)), 1:s_y(3),...
            'UniformOutput', false));
        likelihoods(restart_idx) = sum(p_x(:) .* conditional_likelhoods(:));
    end
    [~, most_likely] = max(likelihoods);
    pi_1 = pi_1(most_likely, :);
    P = P(:, :, most_likely);
    emission_means = emission_means(most_likely, :, :);
    emission_covs = emission_covs(:, :, :, most_likely);

end