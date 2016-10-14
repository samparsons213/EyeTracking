function l = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means, emission_covs,...
    zero_probs)
% Returns the likelihood of the parameters given one person's data and the
% target data

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

% zero_probs:       n by m binary array, where zeros correspond to the
%                   latent directions that have zero posterior probability,
%                   and ones to all other latent directions

% Outputs:

% l:                positive real number giving the log-likelihood of the
%                   current parameters given the data and the targets

% Author:           Sam Parsons
% Date created:     07/10/2016
% Last amended:     07/10/2016

%     *********************************************************************
%     Check input arguments
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
    if ~(isnumeric(emission_covs) && isreal(emission_covs) &&...
            (ndims(emission_covs) == 3) &&...
            all(size(emission_covs) == [2 2 s_y(3)]))
        err_msg = ['emission_covs must be [2 2 size(y, 3)]',...
            ' array of real numbers'];
        error(err_msg)
    end
    pos_def = arrayfun(@(x_idx) all(eig(emission_covs(:, :, x_idx)) > 0),...
        1:s_y(3));
    if ~all(pos_def)
        error('all submatrices of emission_covs must be psoitive definite')
    end
%     zero_probs must be [n m] binary array
    if ~(isnumeric(zero_probs) && isreal(zero_probs) && ismatrix(zero_probs) &&...
            all(size(zero_probs) == [s_y(1), s_y(3) - 1]) &&...
            all((zero_probs(:) == 0) | (zero_probs(:) == 1)))
        error('zero_probs must be a [size(y, 1) size(y, 3)-1] binary array')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Use priorMarginals.m to obtain the prior marginals
%     p(x_t) for each time t. Then calculate the conditional likelihoods
%     p(y_t | x_t) for each time t and state x_t. Then take the average
%     over x_t: p(y_t) = sum_x_t p(x_t)*p(y_t | x_t) for each t. Then take
%     logs and sum over time to get the data log likelihood
%     *********************************************************************

% %     p_x = priorMarginals(pi_1, P, s_y(1));
% %     conditional_likelihoods = cell2mat(arrayfun(@(x_idx)...
% %         mvnpdf(y(:, :, x_idx), emission_means(:, :, x_idx),...
% %         emission_covs(:, :, x_idx)), 1:s_y(3), 'UniformOutput', false));
% %     l = sum(p_x .* conditional_likelihoods, 2);
% %     l = sum(log(l));
    
    yu_conditional_likelihoods = zeros(s_y(1), 1);
    zero_probs = [ones(s_y(1), 1), zero_probs];
    x_conditional_likelihoods = cell2mat(arrayfun(@(x_idx)...
        mvnpdf(y(:, :, x_idx), emission_means(:, :, x_idx),...
        emission_covs(:, :, x_idx)), 1:s_y(3),...
        'UniformOutput', false));
    x_conditional_likelihoods(x_conditional_likelihoods == 0) = realmin;
    x_conditional_likelihoods = x_conditional_likelihoods .*...
         zero_probs;
    alpha_t = pi_1 .* x_conditional_likelihoods(1, :) .* p_ugx(1, :);
    alpha_t(alpha_t == 0) = realmin;
    alpha_t = alpha_t .* zero_probs(1, :);
    yu_conditional_likelihoods(1) = sum(alpha_t);
    filtered_x = alpha_t ./ yu_conditional_likelihoods(1);
    for t = 2:s_y(1)
        alpha_t = (filtered_x * P) .* x_conditional_likelihoods(t, :) .*...
            p_ugx(t, :);
        yu_conditional_likelihoods(t) = max(realmin, sum(alpha_t));
        filtered_x = alpha_t ./ yu_conditional_likelihoods(t);
    end
    l = sum(yu_conditional_likelihoods);

end