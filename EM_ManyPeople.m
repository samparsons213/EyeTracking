function [pi_1, P, emission_means, emission_covs] = EM_ManyPeople(y, p_ugx,...
    pi_1, P, emission_means, emission_covs, zero_probs, n_iters, epsilon,...
    min_eig)
% Performs n_iters iterations of EM on multiple people's data for one
% experiment

% Inputs:

% y:                n_people by 1 cell, each element of which is an n by 2
%                   by (m+1) array of transformed eye tracking difference
%                   data. First element of third dim is untransformed,
%                   corresponding to 'no movement' latent state, remaining
%                   elements are transformed according to corresponding
%                   columns of l_dirs

% p_ugx:            n_people by 1 cell, ecah element of which is an n by
%                   (m+1) array of unnormalised conditional densities of
%                   u_t | x_t for each possible latent state - first
%                   element corresponds to 'no movement' latent state, last
%                   n elements correspond to each direction in l_dirs

% pi_1:             1 by (m+1) probability distribution (sums to 1) giving
%                   the prior distribution of x_1

% P:                (n+1) by (n+1) matrix of transition probabilties, with
%                   each row summing to 1

% emission_means:   1 by 2 by (m+1) array of mean vectors for the Gaussian 
%                   emission densities for each latent state

% emission_covs:    2 by 2 by (m+1) array of covariance matrices for
%                   the Gaussian emission densities for each latent state

% zero_probs:       n_people by 1 cell, each element of which is an n by m
%                   binary array, where zeros correspond to the latent
%                   directions that have zero posterior probability, and
%                   ones to all other latent directions

% n_iters:          positive integer giving the maximum number of
%                   iterations of EM to run

% epsilon:          positive real number giving convergence criteria for
%                   parameters

% min_eig:          positive real number giving the minimum acceptable
%                   eigenvalue  of estimated covariance matrices

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
% Date created:     27/09/2016
% Last amended:     28/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 10
        error('all 10 arguments must be input')
    end
%     y must be a [n_people 1] cell array
    if ~(iscell(y) && iscolumn(y))
        error('y must be a [n_people 1] cell array')
    end
    n_people = length(y);
%     All elements of y must be [n 2 m+1] for some n and common m
    dim_x = arrayfun(@(person_idx) size(y{person_idx}, 3), 1:n_people);
    if ~all(dim_x == dim_x(1))
        err_msg = ['Number of transformations / latent states must be',...
            ' equal across people'];
        error(err_msg)
    end
    dim_x = dim_x(1);
%     p_ugx must be a [n_people 1] cell array
    if ~(iscell(p_ugx) && iscolumn(p_ugx) && (length(p_ugx) == n_people))
        error('p_ugx must be a [length(y) 1] cell array')
    end
%     pi_1 must be [1 m+1] array of non-negative reals that sum to 1
    num_tol = 1e-8;
    if ~(isnumeric(pi_1) && isreal(pi_1) && isrow(pi_1) &&...
            (length(pi_1) == dim_x) && all(pi_1 >= 0) &&...
            (abs(1 - sum(pi_1)) < num_tol))
        error('pi_1 must be [1 dim_x] probability distribution')
    end
%     P must be [m+1 m+1] and each row must be a probability distribution
%     (sums to 1)
    if ~(isnumeric(P) && isreal(P) && ismatrix(P) && all(size(P) == dim_x))
        error('P must be a real square matrix of size dim_x')
    end
    is_prob_dist = arrayfun(@(ld_idx) all(P(ld_idx, :) >= 0) &&...
        (abs(1 - sum(P(ld_idx, :))) < num_tol), 1:dim_x);
    if ~all(is_prob_dist)
        error('all rows of P must be probability distributions')
    end
%     emission_means must be [1 2 m+1] array of real numbers
    if ~(isnumeric(emission_means) && isreal(emission_means) &&...
            (ndims(emission_means) == 3) &&...
            all(size(emission_means) == [1, 2, dim_x]))
        error('emission_means must be a real array of size [1 2 dim_x]')
    end
%     emission_covs must be [2 2 m+1] array of real covariance matrices.
%     Symmetric positive definiteness is tested in forwardBackward.m
    if ~(isnumeric(emission_covs) && isreal(emission_covs) &&...
            (ndims(emission_covs) == 3) &&...
            all(size(emission_covs) == [2 2 dim_x]))
        err_msg = ['emission_covs must be [2 2 dim_x]',...
            ' array of real numbers'];
        error(err_msg)
    end
%     zero_probs must be a [n_people 1] cell array. Checks on each element
%     of zero_probs are performed in forwardBackward.m - must be converted
%     to double in forwardBackward.m
    if ~(iscell(zero_probs) && iscolumn(zero_probs) &&...
            (length(zero_probs) == n_people))
        error('zero_probs must be a [n_people 1] cell array')
    end
%     n_iters must be a positive integer
    if ~(isnumeric(n_iters) && isreal(n_iters) && isscalar(n_iters) &&...
            (n_iters == round(n_iters)) && (n_iters > 0))
        error('n_iters must be a positive integer')
    end
%     epsilon must be a positive real number
    if ~(isnumeric(epsilon) && isreal(epsilon) && isscalar(epsilon) &&...
            (epsilon > 0))
        error('epsilon must be a positive real number')
    end
%     min_eig must be a positive real number
    if ~(isnumeric(min_eig) && isreal(min_eig) && isscalar(min_eig) &&...
            (min_eig > 0))
        error('min_eig must be a positive real number')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of function. Within each EM iteration, a per-person loop
%     operates. Inside this loop, a forward-backward pass is completed,
%     giving the posterior distribution of the latent variables given the
%     observed data and the control sequence for each person. Then (after
%     the per-person loop is copmleted), the posterior is used to optimise
%     the parameters relative to their previous setting. This two-stage
%     procedure is iterated a maximum of n_iters times to produce final
%     estimate of parameters, ending early if maximum parameter change is
%     less than epsilon
%     *********************************************************************

    p_x = cell(n_people, 1);
    p_x_x_next = cell(n_people, 1);
    iter = 0;
    while iter < n_iters
        iter = iter + 1;
        fprintf('EM iteration %d\n', iter)
        old_parms = [pi_1(:); P(:); emission_means(:); emission_covs(:)];
        fb_completed = false;
        while ~fb_completed
            fb_completed = true;
            try
                for person_idx = 1:n_people
                    [p_x{person_idx}, p_x_x_next{person_idx}] =...
                        forwardBackward(y{person_idx}, p_ugx{person_idx},...
                        pi_1, P, emission_means, emission_covs, double(zero_probs{person_idx}));
                end
            catch ME
                fprintf('\tforwardBackward.m failed. Re-initialising emission_covs and trying again\n')
                iter = 1;
                fb_completed = false;
                [pi_1, P, emission_means, emission_covs] =...
                    generateRandomParameters(y, zero_probs);
            end
        end
        [pi_1, P, emission_means, emission_covs] =...
            optimiseParametersManyPeople(y, p_x, p_x_x_next, min_eig);
        mean_delta = mean(abs(old_parms - [pi_1(:); P(:); emission_means(:);...
            emission_covs(:)]));
        max_delta = max(abs(old_parms - [pi_1(:); P(:); emission_means(:);...
            emission_covs(:)]));
        fprintf('\tmean |delta| = %f\tmax |delta| = %f\n', mean_delta,...
            max_delta)
        if max_delta < epsilon
            break
        end
    end

end