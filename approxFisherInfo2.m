function [I] = approxFisherInfo2(pi_1, P, emission_means, chol_emission_covs,...
    target_placement, mu1, chol_Sig1, l_dirs, orth_l_dirs, median_norm_u,...
    movement_z, epsilon, d_ec_inv_d_ec, n_samples)
% Approximates the Fisher Information of the HMM at the given parameter
% settings, using analytical gradients, and Monte Carlo samples to
% approximate the expectation

% Inputs:

% pi_1:             1 by (m+1) probability distribution (sums to 1) giving
%                   the prior distribution of x_1

% P:                (m+1) by (m+1) matrix of transition probabilties, with
%                   each row summing to 1

% emission_means:   1 by 2 by (m+1) array of mean vectors for the Gaussian 
%                   emission densities for each latent state

% chol_emission_covs:    2 by 2 by (m+1) array of cholesky suare roots of
%                        covariance matrices for the Gaussian emission
%                        densities for each latent state

% target_sequence:  (T+1) by 2 array of target placements used to adjust
%                   transition probabilities of latent process

% mu1:              1 by 2 mean vector for eye track location at first
%                   time point (pre-differencing)

% chol_Sig1:        2 by 2 cholesky square root of covariance matrix of eye
%                   track location at first time point (pre-differencing)

% l_dirs:           2 by m array of unit vectors corresponding to the m
%                   'movement' latent states

% orth_l_dirs:      2 by m array of unit vectors, each orthogonal to the
%                   corresponding vector in l_dirs

% median_norm_u:    positive real number giving the median norm of the
%                   control directions in the experiment

% movement_z:       positive real giving the normalising constant for the
%                   conditional distributions p(u | x) for x =
%                   2,...,n_l_dirs. Should equal
%                  integral(@(theta) 2\log(2) - log(1-cos(theta)), 0, 2*pi)
%                  = 6.5328 (4 d.p.)

% epsilon:          real number in (0, 0.5) giving the closest acceptable
%                   norm-difference (see below) to 0 and 1

% d_ec_inv_d_ec:    2 by 2 by 2 by 2 by (m+1) array of the derivatives of
%                   emission_covs_inv(c, d, x) wrt emission_covs(a, b, x)

% n_samples:        positive integer giving number of Monte Carlo samples
%                   of synthetic to draw when approximating expectation

% Outputs:

% I:                n_parms by n_parms psd matrix giving the Fisher
%                   Information

% Author:           Sam Parsons
% Date created:     15/11/2016
% Last amended:     15/11/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 14
        error('all args must be input')
    end
    
%     pi_1 must be a [1 m+1] probability distribution for m >= 1
    num_tol = 1e-8;
    if ~(isnumeric(pi_1) && isreal(pi_1) && isrow(pi_1) &&...
            (length(pi_1) > 1) && all(pi_1 >= 0) &&...
            (abs(1 - sum(pi_1)) < num_tol))
        error('pi_1 must be a [1 m+1] probability distribution')
    end
    dim_x = length(pi_1);
    
%     P must be a [m+1 m+1] matrix, each row of which is a probability
%     distribution
    if ~(isnumeric(P) && isreal(P) && ismatrix(P) && all(size(P) == dim_x))
        error('P must be a [m+1 m+1] matrix')
    end
    is_prob_dist = arrayfun(@(row_idx)...
        all(P(row_idx, :) > 0) && (abs(1 - sum(P(row_idx, :))) < num_tol),...
        1:dim_x);
    if ~all(is_prob_dist)
        error('all rows of P must be probability distributions')
    end
    
%     emission_means must be [1 2 m+1]
    if ~(isnumeric(emission_means) && isreal(emission_means) &&...
            (ndims(emission_means) == 3) &&...
            all(size(emission_means) == [1 2 dim_x]))
        error('emission_means must be [1 2 m+1]')
    end
    
%     chol_emission_covs must be [2 2 m+1] array of cholesky factors
    if ~(isnumeric(chol_emission_covs) && isreal(chol_emission_covs) &&...
            (ndims(chol_emission_covs) == 3) &&...
            all(size(chol_emission_covs) == [2 2 dim_x]))
        error('emission_covs must be [2 2 m+1]')
    end
    is_chol_factor = arrayfun(@(x_idx) all(diag(chol_emission_covs(:, :, x_idx)) > 0) &&...
        (chol_emission_covs(2, 1, x_idx) == 0), 1:dim_x);
    if ~all(is_chol_factor)
        error('all matrices in emission_covs(:, :, x_idx) must be cholesky factors')
    end
    
%     target_placement must be [T+1, 2] for T>=1
    s_tp = size(target_placement);
    if ~(isnumeric(target_placement) && isreal(target_placement) &&...
            ismatrix(target_placement) && (s_tp(2) == 2) && (s_tp(1) > 1))
        error('target_placement must be [T+1, 2] for T>=1')
    end
    
%     mu1 must be [1 2]
    if ~(isnumeric(mu1) && isreal(mu1) && isrow(mu1) && (length(mu1) == 2))
        error('mu1 must be [1 2]')
    end
    
%     chol_Sig1 must be [2 2] cholesky factor
    if ~(isnumeric(chol_Sig1) && isreal(chol_Sig1) && ismatrix(chol_Sig1) &&...
            all(size(chol_Sig1) == [2, 2]) &&...
            all(diag(chol_Sig1) > -num_tol) && (chol_Sig1(2, 1) == 0))
        error('chol_Sig1 must be [2 2] cholesky factor')
    end
    
%     l_dirs must be [2 m] array of unit vectors
    is_unit = arrayfun(@(x_idx) abs(norm(l_dirs(:, x_idx)) - 1) < num_tol,...
        1:(dim_x-1));
    if ~(isnumeric(l_dirs) && isreal(l_dirs) && ismatrix(l_dirs) &&...
            all(size(l_dirs) == [2 (dim_x-1)]) && all(is_unit))
        error('l_dirs must be [2 m] with all columns unit vectors')
    end
    
%     orth_l_dirs must be [2 m] array of unit vectors, with all vectors
%     orthogonal to corresponding col of l_dirs
    is_unit = arrayfun(@(x_idx)...
        abs(norm(orth_l_dirs(:, x_idx)) - 1) < num_tol, 1:(dim_x-1));
    is_orth = arrayfun(@(x_idx)...
        abs(orth_l_dirs(:, x_idx)' * l_dirs(:, x_idx)) < num_tol,...
        1:(dim_x-1));
    if ~(isnumeric(orth_l_dirs) && isreal(orth_l_dirs) && ismatrix(orth_l_dirs) &&...
            all(size(orth_l_dirs) == [2 (dim_x-1)]) && all(is_unit) &&...
            all(is_orth))
        err_msg = ['orth_l_dirs must be [2 m] with all columns unit ',...
            'vectors orthogonal to corresponding col of l_dirs'];
        error(err_msg)
    end

    %     median_norm_u must be a positive real scalar
    if ~(isscalar(median_norm_u) && isnumeric(median_norm_u) &&...
            isreal(median_norm_u) && (median_norm_u > 0))
        error('median_norm_u must be a positive real scalar')
    end
%     movement_z must be a positive real scalar
    if ~(isnumeric(movement_z) && isreal(movement_z) &&...
            isscalar(movement_z) && (movement_z > 0))
        error('movement_z must be a positive real scalar')
    end
%     epsilon must be a real scalar in (0, 0.5)
    if ~(isscalar(epsilon) && isnumeric(epsilon) && isreal(epsilon) &&...
            (epsilon > 0) && (epsilon < 0.5))
        error('epsilon must be a real scalar in (0, 0.5)')
    end
%     n_samples must be a positive integer
    if ~(isnumeric(n_samples) && isreal(n_samples) && isscalar(n_samples) &&...
            (n_samples == round(n_samples)) && (n_samples > 0))
        error('n_samples must be a positive integer')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. First, emission_covs is computed from
%     chol_emission_covs. Then, procedure iteratively draws a sample,
%     computes its approximate score vector, and adjusts the current value
%     of the approximation of the Fisher Info.
%     *********************************************************************

    emission_covs = cell2mat(arrayfun(@(x_idx) chol_emission_covs(:, :, x_idx)' *...
        chol_emission_covs(:, :, x_idx), reshape(1:dim_x, 1, 1, dim_x),...
        'UniformOutput', false));
    
    [y, zero_prob, p_ugx] = oneSyntheticData(pi_1, P, emission_means,...
        chol_emission_covs, target_placement, mu1, chol_Sig1, l_dirs,...
        orth_l_dirs, median_norm_u, movement_z, epsilon);
    grad = gradLogLYU(y, p_ugx, pi_1, P, emission_means,...
        emission_covs, double(zero_prob), d_ec_inv_d_ec);
    I = grad * grad';
    
    for sample = 2:n_samples
        if mod(sample, 500) == 0
            fprintf('Drawing sample no: %d\n', sample)
        end
        [y, zero_prob, p_ugx] = oneSyntheticData(pi_1, P, emission_means,...
            chol_emission_covs, target_placement, mu1, chol_Sig1, l_dirs,...
            orth_l_dirs, median_norm_u, movement_z, epsilon);
        grad = gradLogLYU(y, p_ugx, pi_1, P, emission_means,...
            emission_covs, double(zero_prob), d_ec_inv_d_ec);
        I = ((sample-1)/sample) .* I + (1/sample) .* (grad * grad');
    end

end