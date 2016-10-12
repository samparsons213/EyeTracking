function [emission_means, emission_covs] =...
    initialEmissionParametersManyPeople(y, zero_probs)
% Returns sample mean vectors and covariance matrices of y (transformed
% difference data of eye tracks) of all observations not given zero
% probablity (for having y(t, 1, x_idx) = log(0) or log(-z) for some z>0
% for that t) for each latent state, where y and zero_probs are cell arrays
% containing y and zero_probs respectively, for many people for one
% experiment

% Inputs:

% y:                n_people by 1 cell array, each element of which is an n
%                   by 2 by (m+1) array of transformed eye tracking
%                   difference data. First element of third dim is
%                   untransformed, corresponding to 'no movement' latent
%                   state, remaining elements are transformed according to
%                   corresponding columns of l_dirs

% zero_probs:       n_people by 1 cell array, each element of which is an n
%                   by m binary array, where zeros correspond to the latent
%                   directions that have zero posterior probability, and
%                   ones to all other latent directions

% Outputs:

% emission_means:   1 by 2 by (m+1) array of mean vectors for the Gaussian 
%                   emission densities for each latent state

% emission_covs:    2 by 2 by (m+1) array of covariance matrices for
%                   the Gaussian emission densities for each latent state

% Author:           Sam Parsons
% Date created:     28/09/2016
% Last amended:     28/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 2
        error('all 2 arguments must be input')
    end
%     y must be a [n_people 1] cell array
    if ~(iscell(y) && iscolumn(y))
        error('y must be an [n_people 1] cell array')
    end
%     zero_probs must be an [n_people 1] cell array
    n_people = length(y);
    if ~(iscell(zero_probs) && iscolumn(zero_probs) &&...
            (length(zero_probs) == n_people))
        error('zero_probs must be an [n_people 2] cell array')
    end
%     each element of y must be a [n 2 dim_x] real array, and each element
%     of zero_probs must be [n dim_x] binary array
    dim_x = size(y{1}, 3);
    for person_idx = 1:n_people
        s_y = size(y{person_idx});
%         Check on y
        if ~(isnumeric(y{person_idx}) && isreal(y{person_idx}) &&...
                (ndims(y{person_idx}) == 3) && all(s_y(2:3) == [2, dim_x]))
            error('y must be a [n 2 dim_x] real array')
        end
%         Check on zero_probs
        zero_probs{person_idx} = double(zero_probs{person_idx});
        if ~(isnumeric(zero_probs{person_idx}) &&...
                isreal(zero_probs{person_idx}) &&...
                ismatrix(zero_probs{person_idx}) &&...
                all(size(zero_probs{person_idx}) == [s_y(1), dim_x - 1]) &&...
                all((zero_probs{person_idx}(:) == 0) | (zero_probs{person_idx}(:) == 1)))
            error('zero_probs must be a [size(y, 1) size(y, 3)-1] binary array')
        end
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. For each person, emission_means takes the sum
%     along dim 1 of y, subject to zero_probs = 1 for that row and state
%     combination (zero_probs is augmented with ones for state 1 as the no
%     movement state always has positive probablity). These per-person sums
%     are then added together and divided by the total number of summands
%     they each comprise. Similarly for emission_covs, the average of
%     positive-probablility squared-difference matrices is taken to be the
%     initialising setting, with the average constructed similarly.
%     *********************************************************************

    zero_probs = arrayfun(@(person_idx)...
        reshape([ones(size(y{person_idx}, 1), 1), zero_probs{person_idx}],...
        size(y{person_idx}, 1), 1, dim_x),...
        (1:n_people)', 'UniformOutput', false);
    emission_means = sum(cell2mat(arrayfun(@(person_idx)...
        sum(bsxfun(@times, y{person_idx}, zero_probs{person_idx}), 1),...
        (1:n_people)', 'UniformOutput', false)), 1);
    divisors = reshape(sum(cell2mat(arrayfun(@(person_idx)...
        sum(zero_probs{person_idx}, 1),...
        (1:n_people)', 'UniformOutput', false)), 1), 1, 1, dim_x);
    emission_means = bsxfun(@rdivide, emission_means, divisors);
    diffs = arrayfun(@(person_idx)...
        reshape(num2cell(bsxfun(@minus, y{person_idx}, emission_means), 2),...
        1, 1, size(y{person_idx}, 1), dim_x),...
        (1:n_people)', 'UniformOutput', false);
    diffs_sq = arrayfun(@(person_idx)...
        cell2mat(cellfun(@(diff_vec) diff_vec' * diff_vec, diffs{person_idx},...
        'UniformOutput', false)), (1:n_people)', 'UniformOutput', false);
    zero_probs = arrayfun(@(person_idx)...
        reshape(zero_probs{person_idx}, 1, 1, size(y{person_idx}, 1), dim_x),...
        (1:n_people)', 'UniformOutput', false);
    emission_covs = sum(cell2mat(arrayfun(@(person_idx)...
        squeeze(sum(bsxfun(@times, diffs_sq{person_idx}, zero_probs{person_idx}), 3)),...
        reshape(1:n_people, 1, 1, 1, n_people), 'UniformOutput', false)), 4);
    emission_covs = bsxfun(@rdivide, emission_covs, divisors);

end