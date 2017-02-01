function p_x = forwardOneStep(y, p_x, p_ugx, emission_means,...
    emission_covs, log_cov_dets, zero_probs)
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

    isMatrix = @(x) isnumeric(x) && isreal(x) && ismatrix(x);
    is3DMatrix = @(x) isnumeric(x) && isreal(x) && (ndims(x) == 3);
    
    if nargin < 7
        error('all 8 arguments must be input')
    end
    
    s_y = size(y);
    if ~(is3DMatrix(y) && all([s_y(1), s_y(2)] == [1 2]) && (s_y(3) > 1))
        error('y must be a real array of size [1 2 n+1] for n >= 1')
    end

    num_tol = 1e-8;
    if ~(isMatrix(p_x) && (length(p_x) == s_y(3)))
        error('p_x_prev must be [1 size(y, 3)]')
    end
    if ~(all(p_x >= 0) && (abs(1 - sum(p_x)) < num_tol))
        if all(isnan(p_x))
            p_x = ones(1, s_y(3)) ./ s_y(3);
        else
            error('p_x_prev must be [1 size(y, 3)] probability distribution')
        end
    end

    if ~(isMatrix(p_ugx) && all(size(p_ugx) == [1 s_y(3)]) && all(p_ugx > 0))
        error('p_ugx must be a row vector of positive real numbers with length(p_ugx) == size(y, 3)')
    end
    

    if ~(is3DMatrix(emission_means) && all(size(emission_means) == s_y))
        error('emission_means must be a real array of equal size to y')
    end
    
    if ~(is3DMatrix(emission_covs) && all(size(emission_covs) == [2 2 s_y(3)]))
        error('emission_covs must be [2 2 size(y, 3)] array of real numbers')
    end
    

    if ~(isMatrix(log_cov_dets) && (length(log_cov_dets) == s_y(3)))
        error('log_cov_dets must be [1 size(y, 3)] real array')
    end

    if ~(isMatrix(zero_probs) && (length(zero_probs) == s_y(3)-1) &&...
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
    
    mahanalobis = y - emission_means;
    mahanalobis = arrayfun(@(ld_idx) mahanalobis(:, :, ld_idx) *...
        (emission_covs(:, :, ld_idx) \ mahanalobis(:, :, ld_idx)'),...
        1:s_y(3));

    log_emission_densities = -(mahanalobis + log_cov_dets)./ 2;
    log_p_x = log(p_x) + log(p_ugx) + log_emission_densities;
    log_p_x([1 zero_probs] == 0) = -Inf;
    log_p_x= log_p_x-max(log_p_x);
    p_x = exp(log_p_x);
    p_x = p_x ./ sum(p_x);

end