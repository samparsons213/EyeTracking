function [p_x, p_x_x_next] = forwardBackward(y, p_ugx, pi_1, P,...
    emission_means, emission_covs, zero_probs)
% Completes one full forward-backward pass for a given set of parameters,
% returning the smoothed distributions p(x_t | y_1:T, u_1:T) and
% p(x_t, x_t+1 | y_1:T, u_1:T), for one person in one experiment

% Inputs:

% y:                n by 2 by (m+1) array of transformed eye tracking
%                   difference data. First element of third dim is
%                   untransformed, corresponding to 'no movement' latent
%                   state, remaining elements are transformed according to
%                   corresponding columns of l_dirs

% emission_means:   1 by 2 by (m+1) array of mean vectors for the Gaussian 
%                   emission densities for each latent state

% emission_covs:    2 by 2 by (m+1) array of covariance matrices for
%                   the Gaussian emission densities for each latent state

% zero_probs:       n by m binary array, where zeros correspond to the
%                   latent directions that have zero posterior probability,
%                   and ones to all other latent directions


% p_ugx:            n by (m+1) array of unnormalised conditional
%                   probabilities of u_t | x_t for each possible latent
%                   state - first element corresponds to 'no movement'
%                   latent state, last n elements correspond to each
%                   direction in l_dirs. Since this densities are fixed,
%                   once the probabilites are known, it is all we need.

% pi_1:             1 by (m+1) probability distribution (sums to 1) giving
%                   the prior distribution of x_1

% P:                (m+1) by (m+1) matrix of transition probabilties, with
%                   each row summing to 1


% Outputs:

% p_x:              n by (m+1) array of probability distributions (each row
%                   sums to 1), giving smoothed distributions
%                   p(x_t | y_1:T, u_1:T)

% p_x_x_next:       (m+1) by (m+1) by (n-1) array of probability
%                   distributions (each matrix sums to 1), giving smoothed
%                   distributions p(x_t, x_t+1 | y_1:T, u_1:T)

% Author:           Sam Parsons
% Date created:     22/09/2016
% Last amended:     27/09/2016

%% *********************************************************************
%     Check input arguments
%  *********************************************************************

    isMatrix = @(x) isnumeric(x) && isreal(x) && ismatrix(x);
    is3DMatrix = @(x) isnumeric(x) && isreal(x) && (ndims(x) == 3);

    if nargin < 7
        error('all 7 arguments must be input')
    end
    
    s_y = size(y);
    if ~(is3DMatrix(y) && (s_y(2) == 2))
        error('y must be a [n 2 m+1] real array')
    end
    
    if ~(isMatrix(p_ugx) && all(size(p_ugx) == [s_y(1), s_y(3)]) && all(p_ugx(:) > 0))
        error('p_ugx must be a [size(y, 1) size(y, 3)] array of positive reals')
    end
    
    num_tol = 1e-8;
    if ~(isMatrix(pi_1) && (length(pi_1) == s_y(3)) && all(pi_1 >= 0) &&...
            (abs(1 - sum(pi_1)) < num_tol))
        error('pi_1 must be [1 size(y, 3)] probability distribution')
    end
    
    if ~(isMatrix(P) && all(size(P) == s_y(3)))
        error('P must be a real square matrix of size size(y, 3)')
    end
    
    is_prob_dist = arrayfun(@(ld_idx) all(P(ld_idx, :) >= 0) &&...
        (abs(1 - sum(P(ld_idx, :))) < num_tol), 1:s_y(3));
    if ~all(is_prob_dist)
        error('all rows of P must be probability distributions')
    end
    
    if ~(is3DMatrix(emission_means) && all(size(emission_means) == [1, s_y(2), s_y(3)]))
        error('emission_means must be a real array of equal size to y')
    end
    
    if ~(is3DMatrix(emission_covs) && all(size(emission_covs) == [2 2 s_y(3)]))
        error('emission_covs must be [2 2 size(y, 3)] array of real numbers')
    end
    
    
    spd = arrayfun(@(ld_idx) all(eig(emission_covs(:, :, ld_idx)) > 0), 1:s_y(3));
    if ~all(spd)
        error('all submatrices of emission covs must be symmetric positive definite')
    end
    
    if ~(isMatrix(zero_probs) && all(size(zero_probs) == [s_y(1), s_y(3)-1]) &&...
            all((zero_probs(:) == 0) | (zero_probs(:) == 1)))
        error('zero_probs must be a [size(y, 1) size(y, 3)-1] binary array')
    end


%% *********************************************************************
%     Main body of code. Computes the square roots of the determinants of
%     the emission covariance matices, and then performs the forward pass
%     to get filtered distributions, and then the backward pass to get
%     smoothed distribution.
%% *********************************************************************

    p_x = zeros(s_y(1), s_y(3));
    p_x_x_next = zeros(s_y(3), s_y(3), s_y(1)-1);
    log_cov_dets = arrayfun(@(ld_idx) log(det(emission_covs(:, :, ld_idx))), 1:s_y(3));

    p_x(1, :) = forwardOneStep(y(1, :, :), pi_1, p_ugx(1, :),...
        emission_means, emission_covs, log_cov_dets, zero_probs(1, :));
    
    for t = 2:s_y(1)
        p_x(t, :) = forwardOneStep(y(t, :, :), p_x(t-1, :)*P, p_ugx(t, :),...
            emission_means, emission_covs, log_cov_dets, zero_probs(t, :));
        p_x(t, :) = p_x(t, :) + 1e-6;
        p_x(t, :) = p_x(t, :) ./ sum(p_x(t, :));
    end

    for t = (s_y(1)-1):-1:1
        [p_x(t, :), p_x_x_next(:, :, t)] = backwardOneStep(p_x(t, :), p_x(t+1, :), P);
    end
end