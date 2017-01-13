function grad = gradPosteriorEntropy(pi_1, P, emission_means,...
    emission_covs, y, p_ugx, zero_prob, delta)
% Returns the approximate grad vector of the posterior entropy wrt the
% model parameters, approximated via finite differences.

% Inputs:

% p_x:              T by k array of posterior distributions
%                   p(x_t | Y, U; theta)

% p_x_x_next:       k by k by T-1 array of posterior distributions
%                   p(x_t, x_t+1 | Y, U; theta)

% pi_1:             1 by k prior distribution for x_1

% P:                k by k prior transition matrix

% emission_means:   1 by 2 by k array of mean vectors for each latent state

% emission_covs:    2 by 2 by k array of covariance matrices for each
%                   latent state

% y:                T by 2 by k array of difference vectors, transformed
%                   according to each latent state

% p_ugx:            T by k array of unnormalised conditional
%                   probabilities of u_t | x_t for each possible latent
%                   state - first element corresponds to 'no movement'
%                   latent state, last n elements correspond to each
%                   direction in l_dirs

% zero_prob:        T by (k-1) binary array, where zeros correspond to the
%                   latent directions that have zero posterior probability,
%                   and ones to all other latent directions

% delta:            positive real number giving the size of the finite
%                   difference in each inpute parameter used for
%                   approximating partial derivatives

% Outputs:

% grad:             n_parms by 1 grad vector, order is pi_1(:), P(:),
%                   emission_means(:), ~emission_covs(:) (for each matrix
%                   in emission_covs, only elements 1,2,4 are in grad
%                   vector, as element 3 = element 2)

% Author:           Sam Parsons
% Date created:     08/11/16
% Last amended:     08/11/16

%     *********************************************************************
%     Main body of code. Finite differences are used, with only forward
%     differencing as that is only method that will work on probability
%     vectors with 0 values for some elements (have not encountered a 1
%     value so far). Differences in probabilty vectors are implemented as
%     renormalised additions, calculated such that the effect on the
%     variable in question is a positive shift of delta - so values in
%     other non-zero elements are automatically adjustde to compensate.
%     *********************************************************************

    [~, p_x_x_next] = forwardBackward(y, p_ugx, pi_1, P, emission_means,...
        emission_covs, zero_prob);
    H = posteriorEntropy(p_x_x_next);
    nel = numel(emission_covs);
    ec_idx = sort([1:4:nel, 2:4:nel, 4:4:nel])';
    grad = [pi_1(:); P(:); emission_means(:); emission_covs(ec_idx)];
    
%     pi_1
    for x_idx = 1:length(pi_1)
        pi_1_temp = pi_1;
        delta2 = 1 / (((1 - pi_1(x_idx)) / delta) - 1);
        pi_1_temp(x_idx) = pi_1_temp(x_idx) + delta2;
        pi_1_temp = pi_1_temp ./ sum(pi_1_temp);
        [~, p_x_x_next] = forwardBackward(y, p_ugx, pi_1_temp, P,...
            emission_means, emission_covs, zero_prob);
        H_temp = posteriorEntropy(p_x_x_next);
        grad(x_idx) = (H_temp - H) / delta;
    end
    
%     P
    grad_idx = length(pi_1);
    for x_idx2 = 1:length(pi_1)
        for x_idx1 = 1:length(pi_1)
            P_temp = P;
            delta2 = 1 / (((1 - P(x_idx1, x_idx2)) / delta) - 1);
            P_temp(x_idx1, x_idx2) = P_temp(x_idx1, x_idx2) + delta2;
            P_temp(x_idx1, :) = P_temp(x_idx1, :) ./ sum(P_temp(x_idx1, :));
            [~, p_x_x_next] = forwardBackward(y, p_ugx, pi_1, P_temp,...
                emission_means, emission_covs, zero_prob);
            H_temp = posteriorEntropy(p_x_x_next);
            grad_idx = grad_idx + 1;
            grad(grad_idx) = (H_temp - H) / delta;
        end
    end
    
%     emission_means
    for em_idx = 1:numel(emission_means)
        emission_means_temp = emission_means;
        emission_means_temp(em_idx) = emission_means_temp(em_idx) + delta;
        [~, p_x_x_next] = forwardBackward(y, p_ugx, pi_1, P,...
            emission_means_temp, emission_covs, zero_prob);
        H_temp = posteriorEntropy(p_x_x_next);
        grad(grad_idx + em_idx) = (H_temp - H) / delta;
    end
    
%     emission_covs
    grad_idx = grad_idx + numel(emission_means);
    for ec_idx2 = 1:length(pi_1)
        emission_covs_temp = emission_covs;
        emission_covs_temp(1, 1, ec_idx2) =...
            emission_covs_temp(1, 1, ec_idx2) + delta;
        [~, p_x_x_next] = forwardBackward(y, p_ugx, pi_1, P,...
            emission_means, emission_covs_temp, zero_prob);
        H_temp = posteriorEntropy(p_x_x_next);
        grad_idx = grad_idx + 1;
        grad(grad_idx) = (H_temp - H) / delta;
        
        emission_covs_temp = emission_covs;
        emission_covs_temp(2, 1, ec_idx2) =...
            emission_covs_temp(2, 1, ec_idx2) + delta;
        emission_covs_temp(1, 2, ec_idx2) =...
            emission_covs_temp(2, 1, ec_idx2);
        [~, p_x_x_next] = forwardBackward(y, p_ugx, pi_1, P,...
            emission_means, emission_covs_temp, zero_prob);
        H_temp = posteriorEntropy(p_x_x_next);
        grad_idx = grad_idx + 1;
        grad(grad_idx) = (H_temp - H) / delta;

        emission_covs_temp = emission_covs;
        emission_covs_temp(2, 2, ec_idx2) =...
            emission_covs_temp(2, 2, ec_idx2) + delta;
        [~, p_x_x_next] = forwardBackward(y, p_ugx, pi_1, P,...
            emission_means, emission_covs_temp, zero_prob);
        H_temp = posteriorEntropy(p_x_x_next);
        grad_idx = grad_idx + 1;
        grad(grad_idx) = (H_temp - H) / delta;
    end

end