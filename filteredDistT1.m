function p_x = filteredDistT1(y, pi_1, p_ugx, emission_means,...
    emission_covs, root_cov_dets, zero_probs)
% Returns the filtered distribution p(x_1 | y_1, u_1) for t=1, given the
% data y_1, current parameter values and non-parameteric unnormalised
% conditional probabilities p(u_1 | x_1)

% Inputs:

% y:                1 by 2 by (n+1) array of transformed difference vectors
%                   for eye tracking data, with transformation according to
%                   each of the n latent directions (3rd dim elements
%                   2:end) and no transformation for 'no movement' latent
%                   state (3rd dim 1st element)

% pi_1:             1 by (n+1) probability distribution (sums to 1) giving
%                   the prior distribution of x_1

% p_ugx:            1 by (n+1) array of unnormalised conditional
%                   probabilities of u_1 | x_1 for each possible latent
%                   state - first element corresponds to 'no movement'
%                   latent state, last n elements correspond to each
%                   direction in l_dirs

% emission_means:   1 by 2 by (n+1) array of mean vectors for the Gaussian
%                   emission densities for each latent state

% emission_covs:    2 by 2 by (n+1) array of covariance matrices for the
%                   Gaussian emission densities for each latent state

% zero_probs:       1 by n binary array, where zeros correspond to the
%                   latent directions that have zero posterior probability,
%                   and ones to all other directions

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
    if nargin < 7
        error('all 7 arguments must be input')
    end
%     y must be [1 2 n+1] for n>=1 array of real numbers
    s_y = size(y);
    if ~(isnumeric(y) && isreal(y) && (ndims(y) == 3) &&...
            all([s_y(1), s_y(2)] == [1 2]) && (s_y(3) > 1))
        error('y must be a real array of size [1 2 n+1] for n >= 1')
    end
%     pi_1 must be [1 n+1] array of non-negative reals that sum to 1
    num_tol = 1e-8;
    if ~(isnumeric(pi_1) && isreal(pi_1) && isrow(pi_1) &&...
            (length(pi_1) == s_y(3)) && all(pi_1 >= 0) &&...
            (abs(1 - sum(pi_1)) < num_tol))
        error('pi_1 must be [1 size(y, 3)] probability distribution')
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
%     Main body of function. Calculates emission densities (ignoring
%     constants common to all of them) of the observation for each latent
%     state, then multiplies the prior pi_1 by the emission densities and
%     by p_ugx, and then normalises the result to become a probability
%     distribution. Finally, output is multiplied by [1 zero_probs] so
%     latent directions with zero posterior probability go to 0
%     *********************************************************************

    mahanalobis = y - emission_means;
    mahanalobis = arrayfun(@(ld_idx) mahanalobis(:, :, ld_idx) *...
        (emission_covs(:, :, ld_idx) \ mahanalobis(:, :, ld_idx)'),...
        1:s_y(3));
    emission_densities = exp(-mahanalobis ./ 2) ./ root_cov_dets;
    p_x = pi_1 .* p_ugx .* emission_densities;
    p_x = p_x .* [1 zero_probs];
    p_x = p_x ./ sum(p_x);

end