function [H] = approxHessLogLYU2(y, p_ugx, pi_1, P, emission_means,...
    emission_covs, zero_probs, delta)
% Uses finite differences to approximate the Hessian matrix of the
% log-likeilhood evaluated at the fitted parameters with one person's data
% (including target data)

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

% delta:            positive real number, giving the half-width of the
%                   parameter shift interval in finite differences

% Outputs:

% H:                n_parms by n_parms approximate Hessian matrix as
%                   described above

% Author:           Sam Parsons
% Date created:     11/11/2016
% Last amended:     11/11/2016

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
        error('all submatrices of emission_covs must be positive definite')
    end
%     zero_probs must be [n m] binary array
    if ~(isnumeric(zero_probs) && isreal(zero_probs) && ismatrix(zero_probs) &&...
            all(size(zero_probs) == [s_y(1), s_y(3) - 1]) &&...
            all((zero_probs(:) == 0) | (zero_probs(:) == 1)))
        error('zero_probs must be a [size(y, 1) size(y, 3)-1] binary array')
    end
%     delta must be a positive real number
    if ~(isnumeric(delta) && isreal(delta) && isscalar(delta) &&...
            (delta > 0))
        error('delta must be a positive real number')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of function. Partial differential with respect to each
%     input parameter is approximated one by one, as
%     H_ii = ( l(theta_i++) - 2l(theta_i+) + l(theta) ) / ( delta^2 ),
%     H_ij = ( l(theta_i+j+) - l(theta_i+j) - l(theta_ij+) + l(theta) ) / ( delta^2 ).
%     *********************************************************************
    s_P = size(P);
    s_ec = size(emission_covs);
    n_parms = [numel(pi_1); numel(P); numel(emission_means); 0];
    n_parms(4) = s_ec(3) * s_ec(2) * (s_ec(2)+1) / 2;
    H = zeros(sum(n_parms));
    l = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means, emission_covs,...
        zero_probs);
    grad = approxGradLogLYU(y, p_ugx, pi_1, P, emission_means,...
        emission_covs, zero_probs, delta);
    delta_x2 = 2 * delta;
%     *********************************************************************
%     Diagonal of H
%     *********************************************************************

%     pi_1
    for parm_idx = 1:n_parms(1)
        pi_1_plus = pi_1;
        delta2 = 1 / (((1 - pi_1(parm_idx)) / delta_x2) - 1);
        pi_1_plus(parm_idx) = pi_1_plus(parm_idx) + delta2;
        pi_1_plus = pi_1_plus ./ sum(pi_1_plus);
        l_plus = logLikelihoodYU(y, p_ugx, pi_1_plus, P, emission_means,...
            emission_covs, zero_probs);
        H(parm_idx) = (l_plus - 2*grad(parm_idx) + l) / (delta^2);
    end
%     P
    H_diag_idx = n_parms(1);
    for parm_idx2 = 1:s_P(1)
        for parm_idx1 = 1:s_P(2)
            P_plus = P;
            delta2 = 1 / (((1 - P(parm_idx1, parm_idx2)) / delta_x2) - 1);
            P_plus(parm_idx1, parm_idx2) = P_plus(parm_idx1, parm_idx2) +...
                delta2;
            P_plus(parm_idx1, :) = P_plus(parm_idx1, :) ./...
                sum(P_plus(parm_idx1, :));
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P_plus, emission_means,...
                emission_covs, zero_probs);
            H_diag_idx = H_diag_idx + 1;
            H(H_diag_idx, H_diag_idx) =...
                (l_plus - 2*grad(H_diag_idx) +  l) / (delta^2);
        end
    end
%     emission_means
    for parm_idx = 1:n_parms(3)
        emission_means_plus = emission_means;
        emission_means_plus(parm_idx) = emission_means(parm_idx) + delta_x2;
        l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means_plus,...
            emission_covs, zero_probs);
        H(H_diag_idx + parm_idx, H_diag_idx + parm_idx) =...
            (l_plus - 2*grad(H_diag_idx + parm_idx) + l) / (delta^2);
    end
%     emission_covs
    H_diag_idx = sum(n_parms(1:3));
    for parm_idx1 = 1:s_ec(3)
        emission_covs_plus = emission_covs;
        emission_covs_plus(1, 1, parm_idx1) =...
            emission_covs_plus(1, 1, parm_idx1) + delta_x2;
        l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
            emission_covs_plus, zero_probs);
        H_diag_idx = H_diag_idx + 1;
        H(H_diag_idx, H_diag_idx) =...
            (l_plus - 2*grad(H_diag_idx) + l) / (delta^2);
        
        emission_covs_plus = emission_covs;
        emission_covs_plus(2, 1, parm_idx1) =...
            emission_covs_plus(2, 1, parm_idx1) + delta_x2;
        emission_covs_plus(1, 2, parm_idx1) =...
            emission_covs_plus(1, 2, parm_idx1) + delta_x2;
        l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
            emission_covs_plus, zero_probs);
        H_diag_idx = H_diag_idx + 1;
        H(H_diag_idx, H_diag_idx) =...
            (l_plus - 2*grad(H_diag_idx) + l) / (delta^2);
        
        emission_covs_plus = emission_covs;
        emission_covs_plus(2, 2, parm_idx1) =...
            emission_covs_plus(2, 2, parm_idx1) + delta_x2;
        l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
            emission_covs_plus, zero_probs);
        H_diag_idx = H_diag_idx + 1;
        H(H_diag_idx, H_diag_idx) =...
            (l_plus - 2*grad(H_diag_idx) + l) / (delta^2);
    end
%     *********************************************************************

%     *********************************************************************
%     Off diagonal
%     *********************************************************************

%     dpi_1 d...
    
    for pi_1_idx = 1:(n_parms(1)-1)
%         dpi_1
        for pi_1_idx2 = (pi_1_idx+1):n_parms(1)
            pi_1_plus = pi_1;
            delta21 = 1 / ( (1 - 2*pi_1(pi_1_idx)/(1-pi_1(pi_1_idx2)+pi_1(pi_1_idx)))/delta - 2/(1-pi_1(pi_1_idx2)+pi_1(pi_1_idx)) );
            delta22 = delta21 * (1-pi_1(pi_1_idx)+pi_1(pi_1_idx2)) / (1-pi_1(pi_1_idx2)+pi_1(pi_1_idx));
            pi_1_plus(pi_1_idx) = pi_1_plus(pi_1_idx) + delta21;
            pi_1_plus(pi_1_idx2) = pi_1_plus(pi_1_idx2) + delta22;
            pi_1_plus = pi_1_plus ./ sum(pi_1_plus);
            l_plus = logLikelihoodYU(y, p_ugx, pi_1_plus, P, emission_means,...
                emission_covs, zero_probs);
            H(pi_1_idx, pi_1_idx2) = (l_plus - grad(pi_1_idx) - grad(pi_1_idx2) + l) / (delta^2);
            H(pi_1_idx2, pi_1_idx) = H(pi_1_idx, pi_1_idx2);
        end
        H_idx = n_parms(1);
        delta2 = 1 / (((1 - pi_1(pi_1_idx)) / delta) - 1);
        pi_1_plus = pi_1;
        pi_1_plus(pi_1_idx) = pi_1_plus(pi_1_idx) + delta2;
        pi_1_plus = pi_1_plus ./ sum(pi_1_plus);
%         dP
        for P_idx2 = 1:s_P(2)
            for P_idx1 = 1:s_P(1)
                P_plus = P;
                delta2 = 1 / (((1 - P(P_idx1, P_idx2)) / delta) - 1);
                P_plus(P_idx1, P_idx2) = P_plus(P_idx1, P_idx2) + delta2;
                P_plus(P_idx1, :) = P_plus(P_idx1, :) ./ sum(P_plus(P_idx1, :));
                l_plus = logLikelihoodYU(y, p_ugx, pi_1_plus, P_plus,...
                    emission_means, emission_covs, zero_probs);
                H_idx = H_idx + 1;
                H(pi_1_idx, H_idx) = (l_plus - grad(pi_1_idx) - grad(H_idx) + l) / (delta^2);
                H(H_idx, pi_1_idx) = H(pi_1_idx, H_idx);
            end
        end
%         dem
        for em_idx = 1:n_parms(3)
            em_plus = emission_means;
            em_plus(em_idx) = em_plus(em_idx) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1_plus, P_plus,...
                em_plus, emission_covs, zero_probs);
                H_idx = H_idx + 1;
                H(pi_1_idx, H_idx) = (l_plus - grad(pi_1_idx) - grad(H_idx) + l) / (delta^2);
                H(H_idx, pi_1_idx) = H(pi_1_idx, H_idx);
        end
%         dec
        for ec_idx = 1:n_parms(1)
            ec_plus = emission_covs;
            ec_plus(1, 1, ec_idx) = ec_plus(1, 1, ec_idx) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1_plus, P,...
                emission_means, ec_plus, zero_probs);
                H_idx = H_idx + 1;
                H(pi_1_idx, H_idx) = (l_plus - grad(pi_1_idx) - grad(H_idx) + l) / (delta^2);
                H(H_idx, pi_1_idx) = H(pi_1_idx, H_idx);
            
            ec_plus = emission_covs;
            ec_plus(2, 1, ec_idx) = ec_plus(2, 1, ec_idx) + delta;
            ec_plus(1, 2, ec_idx) = ec_plus(2, 1, ec_idx);
            l_plus = logLikelihoodYU(y, p_ugx, pi_1_plus, P,...
                emission_means, ec_plus, zero_probs);
                H_idx = H_idx + 1;
                H(pi_1_idx, H_idx) = (l_plus - grad(pi_1_idx) - grad(H_idx) + l) / (delta^2);
                H(H_idx, pi_1_idx) = H(pi_1_idx, H_idx);

            ec_plus = emission_covs;
            ec_plus(2, 2, ec_idx) = ec_plus(2, 2, ec_idx) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1_plus, P,...
                emission_means, ec_plus, zero_probs);
                H_idx = H_idx + 1;
                H(pi_1_idx, H_idx) = (l_plus - grad(pi_1_idx) - grad(H_idx) + l) / (delta^2);
                H(H_idx, pi_1_idx) = H(pi_1_idx, H_idx);
        end
    end

%     dP d...
    for P_idx = 1:(s_P(1)*s_P(2)-1)
        H_idx = n_parms(1);
        [P_idx_row, ~] = ind2sub(s_P, P_idx);
        P_plus = P;
        delta2 = 1 / (((1 - P(P_idx)) / delta) - 1);
        P_plus(P_idx)  =P_plus(P_idx) + delta2;
        P_plus(P_idx_row, :) = P_plus(P_idx_row, :) ./ sum(P_plus(P_idx_row, :));
%         dP
        for P_idx2 = (P_idx+1):(s_P(1)*s_P(2))
            [P_idx2_row, ~] = ind2sub(s_P, P_idx2);
            if P_idx_row == P_idx2_row
                delta21 = 1 / ( (1 - 2*P(P_idx)/(1-P(P_idx2)+P(P_idx)))/delta - 2/(1-P(P_idx2)+P(P_idx)) );
                delta22 = delta21 * (1-P(P_idx)+P(P_idx2)) / (1-P(P_idx2)+P(P_idx));
                P_plus2 = P;
                P_plus2(P_idx) = P(P_idx) + delta21;
                P_plus2(P_idx2) = P(P_idx2) + delta22;
                P_plus2(P_idx_row, :) = P_plus2(P_idx_row, :) ./ sum(P_plus2(P_idx_row, :));
                l_plus = logLikelihoodYU(y, p_ugx, pi_1, P_plus2,...
                    emission_means, ec_plus, zero_probs);
                H(H_idx+P_idx, H_idx+P_idx2) = (l_plus - grad(H_idx+P_idx) - grad(H_idx+P_idx2) + l) / (delta^2);
                H(H_idx+P_idx2, H_idx+P_idx) = H(H_idx+P_idx, H_idx+P_idx2);
            else
                delta2 = 1 / (((1 - P(P_idx2)) / delta) - 1);
                P_plus2 = P_plus;
                P_plus2(P_idx2) = P_plus2(P_idx2) + delta2;
                P_plus2(P_idx2_row, :) = P_plus2(P_idx2_row, :) ./ sum(P_plus2(P_idx2_row, :));
                l_plus = logLikelihoodYU(y, p_ugx, pi_1, P_plus2,...
                    emission_means, ec_plus, zero_probs);
                H(H_idx+P_idx, H_idx+P_idx2) = (l_plus - grad(H_idx+P_idx) - grad(H_idx+P_idx2) + l) / (delta^2);
                H(H_idx+P_idx2, H_idx+P_idx) = H(H_idx+P_idx, H_idx+P_idx2);
            end
        end
%         dem
        H_idx = sum(n_parms(1:2));
        for em_idx = 1:n_parms(3)
            em_plus = emission_means;
            em_plus(em_idx) = em_plus(em_idx) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P_plus,...
                em_plus, emission_covs, zero_probs);
                H_idx = H_idx + 1;
                H(n_parms(1)+P_idx, H_idx) = (l_plus - grad(P_idx) - grad(H_idx) + l) / (delta^2);
                H(H_idx, n_parms(1)+P_idx) = H(n_parms(1)+P_idx, H_idx);
        end
%         dec
        for ec_idx = 1:n_parms(1)
            ec_plus = emission_covs;
            ec_plus(1, 1, ec_idx) = ec_plus(1, 1, ec_idx) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P_plus,...
                emission_means, ec_plus, zero_probs);
                H_idx = H_idx + 1;
                H(n_parms(1)+P_idx, H_idx) = (l_plus - grad(pi_1_idx) - grad(H_idx) + l) / (delta^2);
                H(H_idx, n_parms(1)+P_idx) = H(n_parms(1)+P_idx, H_idx);
            
            ec_plus = emission_covs;
            ec_plus(2, 1, ec_idx) = ec_plus(2, 1, ec_idx) + delta;
            ec_plus(1, 2, ec_idx) = ec_plus(2, 1, ec_idx);
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P_plus,...
                emission_means, ec_plus, zero_probs);
                H_idx = H_idx + 1;
                H(n_parms(1)+P_idx, H_idx) = (l_plus - grad(pi_1_idx) - grad(H_idx) + l) / (delta^2);
                H(H_idx, n_parms(1)+P_idx) = H(n_parms(1)+P_idx, H_idx);

            ec_plus = emission_covs;
            ec_plus(2, 2, ec_idx) = ec_plus(2, 2, ec_idx) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P_plus,...
                emission_means, ec_plus, zero_probs);
                H_idx = H_idx + 1;
                H(n_parms(1)+P_idx, H_idx) = (l_plus - grad(pi_1_idx) - grad(H_idx) + l) / (delta^2);
                H(H_idx, n_parms(1)+P_idx) = H(n_parms(1)+P_idx, H_idx);
        end
    end
    
%     dem d...
    H_idx1 = sum(n_parms(1:2));
    for em_idx = 1:(n_parms(3)-1)
        H_idx2 = H_idx1 + 1;
        em_plus = emission_means;
        em_plus(em_idx) = em_plus(em_idx) + delta;
%         dem
        for em_idx2 = (em_idx+1):n_parms(3)
            em_plus2 = em_plus;
            em_plus2(em_idx2) = em_plus2(em_idx2) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, em_plus2,...
                emission_covs, zero_probs);
            H_idx2 = H_idx2 + 1;
            try
                H(H_idx1+em_idx, H_idx2) = (l_plus - grad(H_idx1+em_idx) - grad(H_idx2) + l) / (delta^2);
            catch ME
                rethrow ME
            end
            H(H_idx2, H_idx1+em_idx) = H(H_idx1+em_idx, H_idx2);
        end
%         dec
        for ec_idx = 1:n_parms(1)
            ec_plus = emission_covs;
            ec_plus(1, 1, ec_idx) = ec_plus(1, 1, ec_idx) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, em_plus, ec_plus,...
                zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1+em_idx, H_idx2) = (l_plus - grad(H_idx1+em_idx) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1+em_idx) = H(H_idx1+em_idx, H_idx2);

            ec_plus = emission_covs;
            ec_plus(2, 1, ec_idx) = ec_plus(2, 1, ec_idx) + delta;
            ec_plus(1, 2, ec_idx) = ec_plus(2, 1, ec_idx);
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, em_plus, ec_plus,...
                zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1+em_idx, H_idx2) = (l_plus - grad(H_idx1+em_idx) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1+em_idx) = H(H_idx1+em_idx, H_idx2);

            ec_plus = emission_covs;
            ec_plus(2, 2, ec_idx) = ec_plus(2, 2, ec_idx) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, em_plus, ec_plus,...
                zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1+em_idx, H_idx2) = (l_plus - grad(H_idx1+em_idx) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1+em_idx) = H(H_idx1+em_idx, H_idx2);
        end
    end
    
%     dec d...
    H_idx1 = sum(n_parms(1:3));
    for ec_idx = 1:s_ec(3)
        ec_plus = emission_covs;
        ec_plus(1, 1, ec_idx) = ec_plus(1, 1, ec_idx) + delta;
        H_idx1 = H_idx1 + 1;
        H_idx2 = H_idx1;
%         dec
        ec_plus2 = ec_plus;
        ec_plus2(2, 1, ec_idx) = ec_plus2(2, 1, ec_idx) + delta;
        ec_plus2(1, 2, ec_idx) = ec_plus2(2, 1, ec_idx);
        l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means, ec_plus2,...
            zero_probs);
        H_idx2 = H_idx2 + 1;
        H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
        H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
        
        ec_plus2 = ec_plus;
        ec_plus2(2, 2, ec_idx) = ec_plus2(2, 2, ec_idx) + delta;
        l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means, ec_plus2,...
            zero_probs);
        H_idx2 = H_idx2 + 1;
        H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
        H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
        
        for ec_idx2 = (ec_idx+1):s_ec(3)
            ec_plus2 = ec_plus;
            ec_plus2(1, 1, ec_idx2) = ec_plus2(1, 1, ec_idx2) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
                ec_plus2, zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
            
            ec_plus2 = ec_plus;
            ec_plus2(2, 1, ec_idx2) = ec_plus2(2, 1, ec_idx2) + delta;
            ec_plus2(1, 2, ec_idx2) = ec_plus2(2, 1, ec_idx2);
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
                ec_plus2, zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
            
            ec_plus2 = ec_plus;
            ec_plus2(2, 2, ec_idx2) = ec_plus2(2, 2, ec_idx2) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
                ec_plus2, zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
        end
        
        ec_plus = emission_covs;
        ec_plus(2, 1, ec_idx) = ec_plus(2, 1, ec_idx) + delta;
        ec_plus(1, 2, ec_idx)= ec_plus(2, 1, ec_idx);
        H_idx1 = H_idx1 + 1;
        H_idx2 = H_idx1;
        
        ec_plus2 = ec_plus;
        ec_plus2(2, 2, ec_idx) = ec_plus2(2, 2, ec_idx) + delta;
        l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means, ec_plus2,...
            zero_probs);
        H_idx2 = H_idx2 + 1;
        H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
        H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
        
        for ec_idx2 = (ec_idx+1):s_ec(3)
            ec_plus2 = ec_plus;
            ec_plus2(1, 1, ec_idx2) = ec_plus2(1, 1, ec_idx2) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
                ec_plus2, zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
            
            ec_plus2 = ec_plus;
            ec_plus2(2, 1, ec_idx2) = ec_plus2(2, 1, ec_idx2) + delta;
            ec_plus2(1, 2, ec_idx2) = ec_plus2(2, 1, ec_idx2);
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
                ec_plus2, zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
            
            ec_plus2 = ec_plus;
            ec_plus2(2, 2, ec_idx2) = ec_plus2(2, 2, ec_idx2) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
                ec_plus2, zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
        end
        
        ec_plus = emission_covs;
        ec_plus(2, 2, ec_idx) = ec_plus(2, 2, ec_idx) + delta;
        H_idx1 = H_idx1 + 1;
        H_idx2 = H_idx1;
        
        for ec_idx2 = (ec_idx+1):s_ec(3)
            ec_plus2 = ec_plus;
            ec_plus2(1, 1, ec_idx2) = ec_plus2(1, 1, ec_idx2) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
                ec_plus2, zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
            
            ec_plus2 = ec_plus;
            ec_plus2(2, 1, ec_idx2) = ec_plus2(2, 1, ec_idx2) + delta;
            ec_plus2(1, 2, ec_idx2) = ec_plus2(2, 1, ec_idx2);
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
                ec_plus2, zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
            
            ec_plus2 = ec_plus;
            ec_plus2(2, 2, ec_idx2) = ec_plus2(2, 2, ec_idx2) + delta;
            l_plus = logLikelihoodYU(y, p_ugx, pi_1, P, emission_means,...
                ec_plus2, zero_probs);
            H_idx2 = H_idx2 + 1;
            H(H_idx1, H_idx2) = (l_plus - grad(H_idx1) - grad(H_idx2) + l) / (delta^2);
            H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
        end
    end

end