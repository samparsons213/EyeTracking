function [H] = approxHessLogLYU(y, p_ugx, pi_1, P, emission_means,...
    emission_covs, zero_probs, epsilon)
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

% epsilon:          positive real number, giving the half-width of the
%                   parameter shift interval in finite differences

% Outputs:

% grad:             n_parms by n_parms approximate Hessian matrix as
%                   described above

% Author:           Sam Parsons
% Date created:     26/10/2016
% Last amended:     26/10/2016

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
%     epsilon must be a psoitive real number
    if ~(isnumeric(epsilon) && isreal(epsilon) && isscalar(epsilon) &&...
            (epsilon > 0))
        error('epsilon must be a positive real number')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of function. Partial differential with respect to each
%     input parameter is approximated one by one, as
%     grad_i = ( l(theta_i+) - l(theta_i-) ) / ( 2*epsilon ).
%     *********************************************************************
    s_P = size(P);
    s_ec = size(emission_covs);
    n_parms = [numel(pi_1); numel(P); numel(emission_means);...
        0];
    n_parms(4) = s_ec(3) * s_ec(2) * (s_ec(2)+1) / 2;
    H = zeros(sum(n_parms));
    dbl_eps = 2 * epsilon;
    num_tol = 1e-8;
%     pi_1
    for parm_idx = 1:n_parms(1)
        pi_1_plus = pi_1;
        upshift_star = ((1:n_parms(1)) ~= parm_idx) & (pi_1 > num_tol);
        l_upshift_star = sum(upshift_star);
        upshift = min([epsilon, 1-pi_1(parm_idx), l_upshift_star*min(pi_1(upshift_star))]);
        pi_1_plus(parm_idx) = pi_1(parm_idx) + upshift;
        pi_1_plus(upshift_star) =...
            pi_1_plus(upshift_star) - (upshift/l_upshift_star);
        grad_plus = approxGradLogLYU(y, p_ugx, pi_1_plus, P, emission_means,...
            emission_covs, zero_probs, epsilon);
        pi_1_minus = pi_1;
        downshift_star = ((1:n_parms(1)) ~= parm_idx) & (pi_1 < (1-num_tol));
        l_downshift_star = sum(downshift_star);
        downshift = min([epsilon, pi_1(parm_idx), l_downshift_star*min(1-pi_1(downshift_star))]);
        pi_1_minus(parm_idx) = pi_1(parm_idx) - downshift;
        pi_1_minus(downshift_star) =...
            pi_1_minus(downshift_star) + (downshift/l_downshift_star);
        grad_minus = approxGradLogLYU(y, p_ugx, pi_1_minus, P, emission_means,...
            emission_covs, zero_probs, epsilon);
        H(:, parm_idx) = (grad_plus - grad_minus) ./ (upshift+downshift);
    end
%     P
    H_idx = n_parms(1);
    for parm_idx1 = 1:s_P(1)
        for parm_idx2 = 1:s_P(2)
            P_plus = P;
            upshift_star = ((1:s_P(2)) ~= parm_idx2) & (P(parm_idx1, :) > num_tol);
            l_upshift_star = sum(upshift_star);
            upshift = min([epsilon, 1-P(parm_idx1, parm_idx2), l_upshift_star*min(P(parm_idx1, upshift_star))]);
            P_plus(parm_idx1, parm_idx2) = P(parm_idx1, parm_idx2) + upshift;
            P_plus(parm_idx1, upshift_star) =...
                P_plus(parm_idx1, upshift_star) - (upshift/l_upshift_star);
            grad_plus = approxGradLogLYU(y, p_ugx, pi_1, P_plus, emission_means,...
                emission_covs, zero_probs, epsilon);
            P_minus = P;
            downshift_star = ((1:s_P(2)) ~= parm_idx2) & (P(parm_idx1, :) < (1-num_tol));
            l_downshift_star = sum(downshift_star);
            downshift = min([epsilon, P(parm_idx1, parm_idx2), l_downshift_star*min(1-P(parm_idx1, downshift_star))]);
            P_minus(parm_idx1, parm_idx2) = P(parm_idx1, parm_idx2) - downshift;
            P_minus(parm_idx1, downshift_star) =...
                P_minus(parm_idx1, downshift_star) + (downshift/l_downshift_star);
            grad_minus = approxGradLogLYU(y, p_ugx, pi_1, P_minus, emission_means,...
                emission_covs, zero_probs, epsilon);
            H_idx = H_idx + 1;
            H(:, H_idx) = (grad_plus - grad_minus) ./ (upshift+downshift);
        end
    end
%     emission_means
    for parm_idx = 1:n_parms(3)
        emission_means_plus = emission_means;
        emission_means_plus(parm_idx) = emission_means(parm_idx) + epsilon;
        grad_plus = approxGradLogLYU(y, p_ugx, pi_1, P, emission_means_plus,...
            emission_covs, zero_probs, epsilon);
        emission_means_minus = emission_means;
        emission_means_minus(parm_idx) = emission_means(parm_idx) - epsilon;
        grad_minus = approxGradLogLYU(y, p_ugx, pi_1, P, emission_means_minus,...
            emission_covs, zero_probs, epsilon);
        H(:, H_idx + parm_idx) = (grad_plus - grad_minus) ./ dbl_eps;
    end
%     emission_covs
    H_idx = sum(n_parms(1:3));
    for parm_idx1 = 1:s_ec(3)
        for parm_idx2 = 1:s_ec(2)
            for parm_idx3 = 1:(parm_idx2-1)
                emission_covs_plus = emission_covs;
                emission_covs_plus(parm_idx2, parm_idx3, parm_idx1) =...
                    emission_covs_plus(parm_idx2, parm_idx3, parm_idx1) + epsilon;
                emission_covs_plus(parm_idx3, parm_idx2, parm_idx1) =...
                    emission_covs_plus(parm_idx2, parm_idx3, parm_idx1);
                grad_plus = approxGradLogLYU(y, p_ugx, pi_1, P, emission_means,...
                    emission_covs_plus, zero_probs, epsilon);
                emission_covs_minus = emission_covs;
                emission_covs_minus(parm_idx2, parm_idx3, parm_idx1) =...
                    emission_covs_minus(parm_idx2, parm_idx3, parm_idx1) - epsilon;
                emission_covs_minus(parm_idx3, parm_idx2, parm_idx1) =...
                    emission_covs_minus(parm_idx2, parm_idx3, parm_idx1);
                grad_minus = approxGradLogLYU(y, p_ugx, pi_1, P, emission_means,...
                    emission_covs_minus, zero_probs, epsilon);
                H_idx = H_idx + 1;
                H(:, H_idx) = (grad_plus - grad_minus) ./ dbl_eps;
            end
            emission_covs_plus = emission_covs;
            emission_covs_plus(parm_idx2, parm_idx2, parm_idx1) =...
                emission_covs_plus(parm_idx2, parm_idx2, parm_idx1) + epsilon;
            grad_plus = approxGradLogLYU(y, p_ugx, pi_1, P, emission_means,...
                emission_covs_plus, zero_probs, epsilon);
            emission_covs_minus = emission_covs;
            downshift = min(epsilon, emission_covs(parm_idx2, parm_idx2, parm_idx1) - epsilon);
            emission_covs_minus(parm_idx2, parm_idx2, parm_idx1) =...
                emission_covs_minus(parm_idx2, parm_idx2, parm_idx1) - downshift;
            grad_minus = approxGradLogLYU(y, p_ugx, pi_1, P, emission_means,...
                emission_covs_minus, zero_probs, epsilon);
            H_idx = H_idx + 1;
            H(:, H_idx) = (grad_plus - grad_minus) ./ (epsilon + downshift);
        end
    end

end