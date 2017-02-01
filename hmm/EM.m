function [best_pi_1, best_P, best_emission_means, best_emission_covs, best_log_likelihood] = ...
    EM(y, p_ugx, zero_probs, n_iters, epsilon, min_eig, n_restarts)

% Performs n_iters iterations of EM on one person's data for one experiment

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

% zero_probs:       n by m binary array, where zeros correspond to the
%                   latent directions that have zero posterior probability,
%                   and ones to all other latent directions

% n_iters:          positive integer giving number of iterations of EM to
%                   run

% epsilon:          positive real number giving convergence criteria for
%                   parameters

% min_eig:          positive real number giving the minimum acceptable
%                   eigenvalue  of estimated covariance matrices

% n_restarts:       number of restarts to do in order to find the optimal
%                   parameters

% Outputs:

% pi_1:             1 by (m+1) probability distribution (sums to 1) giving
%                   the prior distribution of x_1

% P:                (n+1) by (n+1) matrix of transition probabilties, with
%                   each row summing to 1

% emission_means:   1 by 2 by (m+1) array of mean vectors for the Gaussian 
%                   emission densities for each latent state

% emission_covs:    2 by 2 by (m+1) array of covariance matrices for
%                   the Gaussian emission densities for each latent state

% Author:           Sam Parsons
% Date created:     22/09/2016
% Last amended:     27/09/2016

% *********************************************************************
%  Check input arguments
% *********************************************************************

    isScalar = @(x) isnumeric(x) && isreal(x) && isscalar(x);
    isPositiveScalar = @(x) isScalar(x) && (x > 0);
    isMatrix = @(x) isnumeric(x) && isreal(x) && ismatrix(x);
    is3DMatrix = @(x) isnumeric(x) && isreal(x) && (ndims(x) == 3);
    isPositiveMatrix = @(x) isnumeric(x) && isreal(x) && ismatrix(x) && all(x(:) > 0);
    
    if nargin < 7
        error('all 7 arguments must be input')
    end

    s_y = size(y);
    
    if ~(isPositiveMatrix(p_ugx) && all(size(p_ugx) == [s_y(1), s_y(3)]))
        error('p_ugx must be a [size(y, 1) size(y, 3)] array of positive reals')
    end

    if ~(isPositiveScalar(n_iters) && (n_iters == round(n_iters)))
        error('n_iters must be a positive integer')
    end
    
    if ~(isPositiveScalar(n_restarts) && (n_restarts == round(n_restarts)))
        error('n_restarts must be a positive integer')
    end
    
    if ~isPositiveScalar(epsilon)
        error('epsilon must be a positive real number')
    end

    if ~isPositiveScalar(min_eig)
        error('min_eig must be a positive real number')
    end
    
% *********************************************************************

% *********************************************************************
%     Main body of function. Within each EM iteration, a forward-backward
%     pass is completed, giving the posterior distribution of the latent
%     variables given the observed data and the control sequence. Then, the
%     posterior is used to optimise the parameters relative to their
%     previous setting. This two-stage procedure is iterated n_iters times
%     to produce final estimate of parameters. The process can be repeated
%     n_restarts to try to find the best random initialisation
% *********************************************************************
    
    zero_probs = double(zero_probs);
    best_log_likelihood = -realmax;
    
    for restart_idx = 1:n_restarts
        fprintf('\n\nStarting parameter estimation step %d...\n\n', restart_idx)
        [pi_1, P, emission_means, emission_covs] = generateRandomParameters(y, zero_probs);
        [pi_1, P, emission_means, emission_covs] = EMoptimiseParameters(y, p_ugx, zero_probs, pi_1, P, ...
            emission_means, emission_covs, n_iters, epsilon, min_eig);
        
        % Keep the best restart
         log_likelihood = dataLogLikelihood(y, p_ugx, zero_probs, pi_1, P, emission_means, emission_covs);
         
         if (log_likelihood > best_log_likelihood),
              [best_pi_1, best_P, best_emission_means, best_emission_covs, best_log_likelihood] = ...
                        deal(pi_1, P, emission_means, emission_covs, log_likelihood);
         end
    end
end

function [pi_1, P, emission_means, emission_covs] = EMoptimiseParameters(y, p_ugx, zero_probs, pi_1, P, ...
            emission_means, emission_covs, n_iters, epsilon, min_eig)
    
    for iter = 1:n_iters
        fprintf('EM iteration %d\n', iter)
        old_parms = [pi_1(:); P(:); emission_means(:); emission_covs(:)];

        % Expectation step
        fb_completed = false;
        while ~fb_completed
            try
                [p_x, p_x_x_next] = forwardBackward(y, p_ugx, pi_1, P, emission_means, emission_covs, zero_probs);
                fb_completed = true;
            catch ME
                warning(getReport(ME))
                warning('\t forwardBackward.m failed. Re-initialising emission_covs and trying again\n')
                [pi_1, P, emission_means, emission_covs] = generateRandomParameters(y, zero_probs);
            end
        end

        % Maximization step
        [pi_1, P, emission_means, emission_covs] = maximizeHMM(y, p_x, p_x_x_next, min_eig);

        % Advance checking
        mean_delta = mean(abs(old_parms - [pi_1(:); P(:); emission_means(:); emission_covs(:)]));
        max_delta = max(abs(old_parms - [pi_1(:); P(:); emission_means(:); emission_covs(:)]));
        fprintf('\tmean |delta| = %f\tmax |delta| = %f\n', mean_delta, max_delta)

        % Note: it is common practice to give n steps of graze
        if max_delta < epsilon
            break
        end
    end
end