function viterbi_path = viterbiPath(y, p_ugx, pi_1, P, emission_means,...
    emission_covs, zero_probs)
% Returns the most likely sequence of latent states for one person's data
% in one experiment, computed as per the Viterbi algorithm.

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

% pi_1:             1 by (m+1) probability distribution (sums to 1) giving
%                   the prior distribution of x_1

% P:                (n+1) by (n+1) matrix of transition probabilties, with
%                   each row summing to 1

% emission_means:   1 by 2 by (m+1) array of mean vectors for the Gaussian 
%                   emission densities for each latent state

% emission_covs:    2 by 2 by (m+1) array of covariance matrices for
%                   the Gaussian emission densities for each latent state

% zero_probs:       n by m logical array, where false corresponds to the
%                   latent directions that have zero posterior probability,
%                   and true to all other latent directions *** Note that
%                   this is a logical array, as opposed to the binary form
%                   it is input in when training ***

% Outputs:

% viterbi_path:     n by 1 integer array, giving the label of the latent
%                   state at each time in the most likely sequence for this
%                   person under the fitted model

% Author:           Sam Parsons
% Date created:     23/09/2016
% Last amended:     23/09/2016

%     *********************************************************************
%     Check inputs
%     *********************************************************************

%     All  arguments must be input
    if nargin < 7
        error('all 7 arguments must be input')
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
%     pi_1 must be [1 m+1] array of non-negative reals that sum to 1
    num_tol = 1e-8;
    if ~(isnumeric(pi_1) && isreal(pi_1) && isrow(pi_1) &&...
            (length(pi_1) == s_y(3)) && all(pi_1 >= 0) &&...
            (abs(1 - sum(pi_1)) < num_tol))
        error('pi_1 must be [1 size(y, 3)] probability distribution')
    end
%     P must be [m+1 m+1] and each row must be a probability distribution
%     (sums to 1)
    if ~(isnumeric(P) && isreal(P) && ismatrix(P) && all(size(P) == s_y(3)))
        error('P must be a real square matrix of size size(y, 3)')
    end
    is_prob_dist = arrayfun(@(ld_idx) all(P(ld_idx, :) >= 0) &&...
        (abs(1 - sum(P(ld_idx, :))) < num_tol), 1:s_y(3));
    if ~all(is_prob_dist)
        error('all rows of P must be probability distributions')
    end
%     emission_means must be [1 2 m+1] array of real numbers
    if ~(isnumeric(emission_means) && isreal(emission_means) &&...
            (ndims(emission_means) == 3) &&...
            all(size(emission_means) == [1, s_y(2), s_y(3)]))
        error('emission_means must be a real array of equal size to y')
    end
%     emission_covs must be [2 2 m+1] array of real covariance matrices.
%     Symmetric positive definiteness is tested in forwardOneStep.m
    if ~(isnumeric(emission_covs) && isreal(emission_covs) &&...
            (ndims(emission_covs) == 3) &&...
            all(size(emission_covs) == [2 2 s_y(3)]))
        err_msg = ['emission_covs must be [2 2 size(y, 3)]',...
            ' array of real numbers'];
        error(err_msg)
    end
    spd = arrayfun(@(ld_idx) issymmetric(emission_covs(:, :, ld_idx)) &&...
        all(eig(emission_covs(:, :, ld_idx)) > 0), 1:s_y(3));
    if ~all(spd)
        err_msg = ['all submatrices of emission covs must be symmetric',...
            ' positive definite'];
        error(err_msg)
    end
%     zero_probs must be [n m+1] binary array
    if ~(islogical(zero_probs) && ismatrix(zero_probs) &&...
            all(size(zero_probs) == [s_y(1), s_y(3) - 1]))
        error('zero_probs must be a [size(y, 1) size(y, 3)-1] logical array')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Performs a forward pass first, recording at each
%     time t a) V(t, i): the log-likelihood of the most likely path up to
%     time t that has x_t = i for each latent state i, and b) W(t-1, i):
%     the latent state at time t-1 that is on this most likely path. V(1,
%     i) is set to log(pi_1(i)) + log(N(Y~_1 | mu^i, Sigma^i)) + 
%     log(P(U_1 | X_1 = i)). Backward pass begins by finding
%     viterbi_path(T) = argmax_i V(T, i) and then recursively computes
%     viterbi_path(t-1) = W(t-1, viterbi_path(t)).
%     *********************************************************************

    log_cov_dets = arrayfun(@(x_idx) log(det(emission_covs(:, :, x_idx))),...
        1:s_y(3));
    diffs = bsxfun(@minus, y, emission_means);
    mahanalobis = arrayfun(@(ld_idx) diffs(1, :, ld_idx) *...
        (emission_covs(:, :, ld_idx) \ diffs(1, :, ld_idx)'),...
        1:s_y(3));
    log_emission_densities = -(mahanalobis + log_cov_dets) ./ 2;
    log_emission_densities(~zero_probs(1, :)) = -Inf;

    log_P = log(P);
    V = log(pi_1) + log(p_ugx(1, :)) + log_emission_densities;
    W = zeros(s_y(1)-1, s_y(3));
    for t = 2:s_y(1)
        mahanalobis = arrayfun(@(ld_idx) diffs(t, :, ld_idx) *...
            (emission_covs(:, :, ld_idx) \ diffs(t, :, ld_idx)'),...
            1:s_y(3));
        log_emission_densities = -(mahanalobis + log_cov_dets) ./ 2;
        log_emission_densities(~zero_probs(t, :)) = -Inf;
        V = log_P + bsxfun(@plus, V', log_emission_densities + p_ugx(t, :));
        [V, W(t-1, :)] = max(V, [], 1);
    end
    
    viterbi_path = zeros(s_y(1), 1);
    [~, viterbi_path(end)] = max(V);
    for t = (s_y(1)-1):-1:1
        viterbi_path(t) = W(t, viterbi_path(t+1));
    end

end