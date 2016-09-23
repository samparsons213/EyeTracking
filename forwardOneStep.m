function p_x = forwardOneStep(y, p_x_prev, P, p_ugx, emission_means,...
    emission_covs, root_cov_dets, zero_probs)
% Performs both the forward predict and forward update of one step of the
% forward pass, returning p(x_t | y_1:t, u_1:t)

% Inputs:

% y:                1 by 2 by (n+1) array of transformed difference vectors
%                   for eye tracking data, with transformation according to
%                   each of the n latent directions (3rd dim elements
%                   2:end) and no transformation for 'no movement' latent
%                   state (3rd dim 1st element)

% p_x_prev:         1 by (n+1) probability distribution (sums to 1) giving
%                   the filtered distribution of x_t-1 | y_1:t-1, u_1:t-1

% P:                (n+1) by (n+1) matrix of transition probabilties, with
%                   each row summing to 1

% p_ugx:            1 by (n+1) array of unnormalised conditional
%                   probabilities of u_t | x_t for each possible latent
%                   state - first element corresponds to 'no movement'
%                   latent state, last n elements correspond to each
%                   direction in l_dirs

% emission_means:   1 by 2 by (n+1) array of mean vectors for the Gaussian
%                   emission densities for each latent state

% emission_covs:    2 by 2 by (n+1) array of covariance matrices for the
%                   Gaussian emission densities for each latent state

% zero_probs:       1 by n binary array, where zeros correspond to the
%                   latent directions that have zero posterior probability,
%                   and ones to all other latent directions

% Outputs:

% p_x:              1 by (n+1) probability distribution (sums to 1) giving
%                   filtered probability distribution described above

% Author:       Sam Parsons
% Date created: 21/09/2016
% Last amended: 21/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 8
        error('all 8 arguments must be input')
    end
%     y must be [1 2 n+1] for n>=1 array of real numbers
    s_y = size(y);
    if ~(isnumeric(y) && isreal(y) && (ndims(y) == 3) &&...
            all([s_y(1), s_y(2)] == [1 2]) && (s_y(3) > 1))
        error('y must be a real array of size [1 2 n+1] for n >= 1')
    end
%     p_x_prev must be [1 n+1] array of non-negative reals that sum to 1
    num_tol = 1e-8;
    if ~(isnumeric(p_x_prev) && isreal(p_x_prev) && isrow(p_x_prev) &&...
            (length(p_x_prev) == s_y(3)) && all(p_x_prev >= 0) &&...
            (abs(1 - sum(p_x_prev)) < num_tol))
        error('p_x_prev must be [1 size(y, 3)] probability distribution')
    end
%     P must be [n+1 n+1] and each row must be a probability distribution
%     (sums to 1). Probability condition checked in forwardBackward.m
    if ~(isnumeric(P) && isreal(P) && ismatrix(P) && all(size(P) == s_y(3)))
        error('P must be a real square matrix of size size(y, 3)')
    end
%     p_ugx must be [1 n+1] array of positive real numbers
    if ~(isnumeric(p_ugx) && isreal(p_ugx) && isrow(p_ugx) &&...
            all(size(p_ugx) == [1 s_y(3)]) && all(p_ugx > 0))
        err_msg = ['p_ugx must be a row vector of positive real numbers',...
            ' with length(p_ugx) == size(y, 3)'];
        error(err_msg)
    end
%     emission_means must be [1 2 n+1] array of real numbers
    if ~(isnumeric(emission_means) && isreal(emission_means) &&...
            (ndims(emission_means) == 3) &&...
            all(size(emission_means) == s_y))
        error('emission_means must be a real array of equal size to y')
    end
%     emission_covs must be [2 2 n+1] array of real covariance matrices.
%     Covariance conditions (symmetric positive definiteness) checked in
%     forwardBackward.m
    if ~(isnumeric(emission_covs) && isreal(emission_covs) &&...
            (ndims(emission_covs) == 3) &&...
            all(size(emission_covs) == [2 2 s_y(3)]))
        error('emission_covs must be [2 2 size(y, 3)] array of real numbers')
    end
%     root_cov_dets must be a [1 n] real array. Note they must also be
%     positive, but this will be managed in forwardBackward.m
    if ~(isnumeric(root_cov_dets) && isreal(root_cov_dets) &&...
            isrow(root_cov_dets) && (length(root_cov_dets) == s_y(3)))
        error('root_cov_dets must be [1 size(y, 3)] real array')
    end
%     zero_probs must be [1 n] binary array
    if ~(isnumeric(zero_probs) && isreal(zero_probs) && isrow(zero_probs) &&...
            (length(zero_probs) == s_y(3) - 1) &&...
            all((zero_probs == 0) | (zero_probs == 1)))
        error('zero_probs must be a [1 size(y, 3)-1] binary array')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of function. Performs predict step as simple matrix
%     multiplication. Then performs update step by multiplying output from
%     predict step with p_ugx and emission densities, and then normalising.
%     Finally, output is multiplied by [1 zero_probs] so latent directions
%     with zero posterior probability go to 0
%     *********************************************************************

    p_x = p_x_prev * P;
    mahanalobis = y - emission_means;
    mahanalobis = arrayfun(@(ld_idx) mahanalobis(:, :, ld_idx) *...
        (emission_covs(:, :, ld_idx) \ mahanalobis(:, :, ld_idx)'),...
        1:s_y(3));
    emission_densities = exp(-mahanalobis ./ 2) ./ root_cov_dets;
    p_x = p_x .* p_ugx .* emission_densities;
    p_x  = p_x .* [1 zero_probs];
    p_x = p_x ./ sum(p_x);

end