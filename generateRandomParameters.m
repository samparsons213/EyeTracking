function [pi_1, P, emission_means, emission_covs] =...
    generateRandomParameters(y, zero_probs, alpha)
% Returns randomly initialised parameters for the proposed HMM, taking one
% hyperparameter as input for producing random probability distributions.
% emission_means and emission_covs are drawn from distributions depending
% on y and zero_probs, for details read below

% Inputs:

% y:                either a [n 2 m+1] array of transformed difference data
%                   for one person and one experiment, or an [n_people 1]
%                   cell array, each element of which is a [n 2 m+1] array
%                   of transformed difference data, for different n but
%                   constant m

% zero_probs:       either a [n 2 m] binary array where zeros indicate zero
%                   posterior probability for that latent state at that
%                   time, and ones indicate positive posterior probability,
%                   or a [n_people 1] cell array, each element of which is
%                   such a binary array with n equal to the corresponding
%                   n in y and constant m, or a logical array of that size
%                   with false <=> 0 and true <=> 1

% alpha:            positive real number giving the single Dirichlet
%                   parameter for all randomly drawn distributions, i.e. if
%                   p is a randomly drawn 1 by n distribution it is sampled
%                   from D(alpha*ones(1, n)). If not input then it is set
%                   to its default value of 1

% Outputs:

% pi_1:             1 by (m+1) latent prior for x_1

% P:                (m+1) by (m+1) latent transition matrix, with row a
%                   probability distribution

% emission_means:   1 by 2 by (m+1) array of emission mean vectors

% emission_covs:    2 by 2 by (m+1) array of emission covariance matrices

% Author:           Sam Parsons
% Date created:     23/09/2016
% Last amended:     29/09/2016

%     *********************************************************************
%     Check inputs
%     *********************************************************************

%     y and zero_probs must be input
    if nargin < 2
        error('y and zero_probs must be input')
    end
%     y must be either a [n 2 m+1] real array, or a [n_people 1] cell
%     array, each element of which is [n 2 m+1] for some m and constant m
    if ~( (iscell(y) && iscolumn(y)) ||...
            (isnumeric(y) && isreal(y) && (ndims(y) == 3) && (size(y, 2) == 2)) )
        err_msg = ['y must either be a [n 2 m+1] reall array, or a',...
            ' [n_people 1] cell array'];
        error(err_msg)
    end
    if iscell(y)
        n_people = length(y);
        dim_x = size(y{1}, 3);
        correct = arrayfun(@(person_idx) isnumeric(y{person_idx}) &&...
            isreal(y{person_idx}) && (ndims(y{person_idx}) == 3) &&...
            all([size(y{person_idx}, 2), size(y{person_idx}, 3)] == [2, dim_x]),...
            1:n_people);
        if ~all(correct)
            err_msg = ['each element in y must be a [n 2 m+1] for some n',...
                ' and constant m'];
            error(err_msg)
        end
    else
        dim_x = size(y, 3);
    end
%     zero_probs must have same form as y; if y is real [n 2 m+1] then
%     zero_probs must be [n m], if y is [n_people 1] cell then zero_probs
%     must be [n_people 1] cell with each element [n m] for same n as
%     corresponding element of y, and either be binary or logical
    if iscell(y)
        if ~(iscell(zero_probs) && iscolumn(zero_probs))
            error('zero_probs must be a [n_people 1] cell array')
        end
        correct_size = arrayfun(@(person_idx) ismatrix(zero_probs{person_idx}) &&...
            all([size(zero_probs{person_idx}, 1), size(zero_probs{person_idx}, 2)] ==...
            [size(y{person_idx}, 1), size(y{person_idx}, 3) - 1]), 1:n_people);
        correct_type = arrayfun(@(person_idx) islogical(zero_probs{person_idx}) ||...
            (isnumeric(zero_probs{person_idx}) &&...
            all((zero_probs{person_idx}(:) == 0) | (zero_probs{person_idx} == 1))),...
            1:n_people);
        if ~all(correct_size & correct_type)
            err_msg = ['zero_probs must be a [n_people 1] cell array, ',...
                'each element of which must be [n m] with n same as',...
                ' corresponding element of y and m constant, and must',...
                ' either be logical or binary'];
            error(err_msg)
        end
    else
        if ~(ismatrix(zero_probs) && (islogical(zero_probs) ||...
                (isnumeric(zero_probs) &&...
                all((zero_probs(:) == 0) | (zero_probs(:) == 1)))))
            error('zero_probs must be [n m] and either be logical or binary')
        end
    end
%     if input, alpha must be a positive real, if not input it is set to 1
    if nargin == 3
        if ~(isnumeric(alpha) && isreal(alpha) && isscalar(alpha) &&...
                (alpha > 0))
            error('alpha must be a positive real number')
        end
    else
        alpha = 1;
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. pi_1 is drawn from D(alpha.*ones(1, dim_x)), as is
%     each row of P. emission_means has all elements drawn from N(0, 1).
%     emission_means and emission_covs are first initialised using either
%     initialEmissionParameters.m or initialEmissionParametersManyPeople.m.
%     Each latent state i's emission_mean is then drawn from
%     N(emission_means(:, :, i), emission_covs(:, :, i)), and
%     emission_covs(:, :, i) is drawn from 
%     Wishart_2(2, emission_covs(:, :, i) ./ 2). The rcond of each
%     generated covariance matrix is checked, and if it's below rcond_tol
%     (currently set to 1e-6), another random matrix is drawn
%     *********************************************************************

    n = 4;
    random_gammas = gamrnd(alpha, 1, (dim_x+1), dim_x);
    pi_1 = random_gammas(1, :) ./ sum(random_gammas(1, :));
    P = bsxfun(@rdivide, random_gammas(2:end, :),...
        sum(random_gammas(2:end, :), 2));
    if iscell(y)
        [emission_means, emission_covs] =...
            initialEmissionParametersManyPeople(y, zero_probs);
    else
        [emission_means, emission_covs] =...
            initialEmissionParameters(y, zero_probs);
    end
    rcond_tol = 1e-6;
    for x_idx = 1:dim_x
        emission_chol_cov = chol(emission_covs(:, :, x_idx));
        emission_means(:, :, x_idx) = emission_means(:, :, x_idx) +...
            randn(1, 2) * emission_chol_cov;
        invertible = false;
        while ~invertible
            emission_cov_temp = randn(n, 2) * (emission_chol_cov ./ sqrt(n));
            emission_cov_temp = emission_cov_temp' * emission_cov_temp;
            if rcond(emission_cov_temp) > rcond_tol
                invertible = true;
                emission_covs(:, :, x_idx) = emission_cov_temp;
            end
        end
    end

end