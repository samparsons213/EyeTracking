function [y, zero_prob, p_ugx, x] = oneSyntheticData(pi_1, P,...
    emission_means, chol_emission_covs, target_placement, mu1, chol_Sig1,...
    l_dirs, orth_l_dirs, median_norm_u, movement_z, epsilon)
% Creates one synthetic datapoint, given the parameters and a sequence of
% target placements

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

% Outputs:

% y:                T by 2 by (m+1) array of transformed difference vectors

% zero_prob:        T by m logical array, where false in position (t, d)
%                   corresponds to zero probability density for the
%                   difference vector at time t being generated from
%                   'movement 'latent state d

% p_ugx:            T by (m+1) array of densities p(u_t | x_t = d) for
%                   position (t, d) for all latent states

% x:                T by 1 array giving the latent state used to draw each
%                   observation

% Author:           Sam Parsons
% Date created:     14/11/2016
% Last amended:     15/11/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 12
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
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Draws first eye track from Gaussian with
%     parameters mu1, Sig1. Then p_ugx(1, :) is calculated and the first
%     latent state is drawn using pi_1 adjusted by p_ugx(1, :). An
%     observation for t=1 is then drawn from the appropriate emission
%     density. Then, for each t in 2:T, p_ugx is calculated and used to
%     adjust the transition matrix P and then draw the next latent state.
%     From these latent states the next observations are drawn before
%     movingd to the next time step. After all observations have been
%     drawn, their transformations needed for inference are calculated.
%     *********************************************************************

    eye_pos = randn(1, 2) * chol_Sig1 + mu1;
    T = s_tp(1) - 1;
    y = zeros(T, 2, dim_x);
    p_ugx = zeros(T, dim_x);
    x = zeros(T, 1);
    l_dirs_t = l_dirs';
    orth_l_dirs_t = orth_l_dirs';
    r = rand(T, 1);
    
    u = target_placement(2, :) - eye_pos;
    p_ugx(1, :) = condProbUX(u', l_dirs, median_norm_u, movement_z, epsilon);
    pi_1_adj = pi_1 .* p_ugx(1, :);
    pi_1_adj = pi_1_adj ./ sum(pi_1_adj);
    pi_1_adj = cumsum(pi_1_adj);
    x(1) = find(r(1) < pi_1_adj, 1);
    y_trans = randn(1, 2) * chol_emission_covs(:, :, x(1)) +...
        emission_means(:, :, x(1));
    if x(1) == 1
        y(1, :, 1) = y_trans;
    else
        y(1, :, 1) = exp(y_trans(1)) .* l_dirs_t(x(1)-1, :) +...
            y_trans(2) .* orth_l_dirs_t(x(1)-1, :);
    end
    eye_pos = eye_pos + y(1, :, 1);
    
    for t = 2:T
        u = target_placement(t+1, :) - eye_pos;
        p_ugx(t, :) = condProbUX(u', l_dirs, median_norm_u, movement_z,...
            epsilon);
        P_adj = P(x(t-1), :) .* p_ugx(t, :);
        P_adj = P_adj ./ sum(P_adj);
        P_adj = cumsum(P_adj);
        x(t) = find(r(t) < P_adj, 1);
        y_trans = randn(1, 2) * chol_emission_covs(:, :, x(t)) +...
            emission_means(:, :, x(t));
        if x(t) == 1
            y(t, :, 1) = y_trans;
        else
            y(t, :, 1) = exp(y_trans(1)) .* l_dirs_t(x(t)-1, :) +...
                y_trans(2) .* orth_l_dirs_t(x(t)-1, :);
        end
        eye_pos = eye_pos + y(t, :, 1);
    end
    
    y_trans1 = y(:, :, 1) * l_dirs;
    zero_prob = y_trans1 > 0;
    y_trans1(zero_prob) = log(y_trans1(zero_prob));
    y(:, 1, 2:end) = reshape(y_trans1, T, 1, dim_x-1);
    y(:, 2, 2:end) = reshape(y(:, :, 1) * orth_l_dirs, T, 1, dim_x-1);

end