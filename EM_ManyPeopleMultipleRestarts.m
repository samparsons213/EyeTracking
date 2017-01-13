function [pi_1, P, emission_means, emission_covs] =...
    EM_ManyPeopleMultipleRestarts(y, p_ugx, zero_probs, n_iters, epsilon,...
    n_restarts, min_eig)
% Performs n_iters iterations of EM on one person's data for one experiment

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

% zero_probs:       n_people by 1 cell, each element of which is an n by m
%                   binary array, where zeros correspond to the latent
%                   directions that have zero posterior probability, and
%                   ones to all other latent directions

% n_iters:          positive integer giving number of iterations of EM to
%                   run

% epsilon:          positive real number giving convergence criteria for
%                   parameters

% n_restarts:       positive integer giving number of restarts of EM
%                   procedure

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
% Date created:     28/09/2016
% Last amended:     30/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 7
        error('all 7 arguments must be input')
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
%     Main body of code. Runs EM_ManyPeople.m n_restarts times, and selects
%     the most likely parameters from those found at each restart.
%     *********************************************************************

    log_likelihoods = zeros(n_people, n_restarts);
    pi_1 = zeros(n_restarts, dim_x);
    P = zeros(dim_x, dim_x, n_restarts);
    emission_means = zeros(n_restarts, 2, dim_x);
    emission_covs = zeros(2, 2, dim_x, n_restarts);
% %     T = arrayfun(@(person_idx) size(y{person_idx}, 1), 1:n_people);
% %     max_T = max(T);
    for restart_idx = 1:n_restarts
        fprintf('\n\nStarting parameter estimation procedure %d...\n\n',...
            restart_idx)
        [pi_1(restart_idx, :), P(:, :, restart_idx),...
            emission_means(restart_idx, :, :),...
            emission_covs(:, :, :, restart_idx)] =...
            generateRandomParameters(y, zero_probs);
        [pi_1(restart_idx, :), P(:, :, restart_idx),...
            emission_means(restart_idx, :, :),...
            emission_covs(:, :, :, restart_idx)] = EM_ManyPeople(y, p_ugx,...
            pi_1(restart_idx, :), P(:, :, restart_idx),...
            emission_means(restart_idx, :, :),...
            emission_covs(:, :, :, restart_idx), zero_probs, n_iters,...
            epsilon, min_eig);
% %         p_x = priorMarginals(pi_1(restart_idx, :), P(:, :, restart_idx),...
% %             max_T);
        for person_idx = 1:n_people
            log_likelihoods(person_idx, restart_idx) =...
                logLikelihoodYU(y{person_idx}, p_ugx{person_idx}, pi_1(restart_idx, :),...
                P(:, :, restart_idx), emission_means(restart_idx, :, :),...
                emission_covs(:, :, :, restart_idx),...
                double(zero_probs{person_idx}));
% %             conditional_likelihoods =...
% %                 cell2mat(arrayfun(@(x_idx) mvnpdf(y{person_idx}(:, :, x_idx),...
% %                 emission_means(restart_idx, :, x_idx),...
% %                 emission_covs(:, :, x_idx, restart_idx)), 1:dim_x,...
% %                 'UniformOutput', false));
% %             p_x_person = p_x(1:T(person_idx), :);
% %             likelihoods(person_idx, restart_idx) =...
% %                 sum(p_x_person(:) .* conditional_likelihoods(:));
        end
    end
    [~, most_likely] = max(sum(log_likelihoods, 1));
    pi_1 = pi_1(most_likely, :);
    P = P(:, :, most_likely);
    emission_means = emission_means(most_likely, :, :);
    emission_covs = emission_covs(:, :, :, most_likely);

end