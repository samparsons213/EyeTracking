function [grad] = gradLogLYU(y, p_ugx, pi_1, P, emission_means,...
    emission_covs, zero_prob, d_ec_inv_d_ec)
% Computes the score vector for one person in on experiment evaluated at
% the fitted parameters.

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

% zero_prob:        n by m binary array, where zeros correspond to the
%                   latent directions that have zero posterior probability,
%                   and ones to all other latent directions

% d_ec_inv_d_ec:    2 by 2 by 2 by 2 by (m+1) array of the derivatives of
%                   emission_covs_inv(c, d, x) wrt emission_covs(a, b, x)

% Outputs:

% grad:             n_parms by 1 approximate grad vector as described
%                   above. Order of elements is pi_1(:), P(:),
%                   emission_means(:), emission_covs(1,2,4)(:)

% Author:           Sam Parsons
% Date created:     25/11/2016
% Last amended:     25/11/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 8
        error('all 8 arguments must be input')
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
        error('all submatrices of emission_covs must be positive definite')
    end
%     zero_prob must be [n m] binary array
    if ~(isnumeric(zero_prob) && isreal(zero_prob) && ismatrix(zero_prob) &&...
            all(size(zero_prob) == [s_y(1), s_y(3) - 1]) &&...
            all((zero_prob(:) == 0) | (zero_prob(:) == 1)))
        error('zero_probs must be a [size(y, 1) size(y, 3)-1] binary array')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Implements grad vector as derived by hand, as
%     efficiently as possible. Derivation is based on derivations in JST's
%     book. Posterior distribution of X|Y,U is first computed. Grad
%     elements for pi_1 and P are trivial to compute once posterior is
%     available. Elements for emission_means are relatively simple,
%     emission_covs elements need to use chain rule over emission_covs_inv
%     and sum out.
%     *********************************************************************

    [p_x, p_x_x_next] = forwardBackward(y, p_ugx, pi_1, P, emission_means,...
        emission_covs, zero_prob);
%     grad_pi_1 and grad_P
    pi_1(pi_1 < num_tol) = num_tol;
    grad_pi_1 = (p_x(1, :) ./ pi_1) - 1;
    s_p_x_x_next_3 = sum(p_x_x_next, 3);
    P(P < num_tol) = num_tol;
    grad_P = bsxfun(@minus, s_p_x_x_next_3 ./ P, sum(s_p_x_x_next_3, 2));
%     grad_emission_means
    y_centred = bsxfun(@minus, y, emission_means);
    [T, m] = size(p_x);
    grad_temp1 = sum(bsxfun(@times, y_centred, reshape(p_x, T, 1, m)), 1);
    grad_em = -cell2mat(arrayfun(@(x_idx)...
        grad_temp1(:, :, x_idx) / emission_covs(:, :, x_idx),...
        1:m, 'UniformOutput', false)) ./ 2;
%     grad_emission_covs
    y_centred_sq = cell2mat(arrayfun(@(t) cell2mat(arrayfun(@(x_idx)...
        y_centred(t, :, x_idx)' * y_centred(t, :, x_idx),...
        reshape(1:m, 1, 1, m), 'UniformOutput', false)),...
        reshape(1:T, 1, 1, 1, T), 'UniformOutput', false));
    grad_temp2 = bsxfun(@minus,...
        y_centred_sq, reshape(emission_covs, 2, 2, m, 1));
    grad_temp3 = bsxfun(@times, reshape(grad_temp2, 2, 2, 1, 1, m, T),...
        reshape(d_ec_inv_d_ec, 2, 2, 2, 2, m, 1));
    grad_temp4 = squeeze(sum(sum(grad_temp3, 1), 2));
    grad_ec =...
        -sum(bsxfun(@times, grad_temp4, reshape(p_x', 1, 1, m, T)), 4) ./ 2;
    n_ec = numel(emission_covs);
    grad_ec_idx = sort([1:4:n_ec, 2:4:n_ec, 4:4:n_ec])';
    
    grad = [grad_pi_1(:); grad_P(:); grad_em(:); grad_ec(grad_ec_idx)];

end