function [emission_means, emission_covs] = initialEmissionParameters(y, zero_probs)
% Returns sample mean vectors and covariance matrices of y (transformed
% difference data of eye tracks) of all observations not given zero
% probablity (for having y(t, 1, x_idx) = log(0) or log(-z) for some z>0
% for that t) for each latent state, where y and zero_probs are cell arrays
% containing y and zero_probs respectively, for many people for one
% experiment

% Inputs:

% y:                n_trials by 1 cell array, each element of which is an n
%                   by 2 by (m+1) array of transformed eye tracking
%                   difference data. First element of third dim is
%                   untransformed, corresponding to 'no movement' latent
%                   state, remaining elements are transformed according to
%                   corresponding columns of l_dirs

% zero_probs:       n_trials by 1 cell array, each element of which is an n
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

    isMatrix = @(x) isnumeric(x) && isreal(x) && ismatrix(x);
    is3DMatrix = @(x) isnumeric(x) && isreal(x) && (ndims(x) == 3);
    isBinaryMatrix = @(x) isMatrix(x) && all((x(:) == 0) | (x(:) == 1));
    
    if nargin < 2
        error('all 2 arguments must be input')
    end

    if ~(iscell(y) && iscolumn(y))
        error('y must be an [n_trials 1] cell array')
    end

    n_people = length(y);
    if ~(iscell(zero_probs) && iscolumn(zero_probs) &&...
            (length(zero_probs) == n_people))
        error('zero_probs must be an [n_trials 2] cell array')
    end

    dim_x = size(y{1}, 3);    
    for person_idx = 1:n_people
        s_y = size(y{person_idx});
        % Check on y
        if ~(is3DMatrix(y{person_idx}) && all(s_y(2:3) == [2, dim_x]))
            error('y must be a [n 2 dim_x] real array')
        end
        % Check on zero_probs
        zero_probs{person_idx} = double(zero_probs{person_idx});
        if ~(isBinaryMatrix(zero_probs{person_idx}) && ...
                all(size(zero_probs{person_idx}) == [s_y(1), dim_x - 1]))
            error('zero_probs must be a [size(y, 1) size(y, 3)-1] binary array')
        end
    end
    
%     *********************************************************************

%     *********************************************************************
%     Main body of code. For each person, emission_means takes the sum
%     along dim 1 of y, subject to zero_probs = 1 for that row and state
%     combination (zero_probs is augmented with ones for state 1 as the no
%     movement state always has positive probablity). These per-person sums
%     are then weighted averaged together by the total number of summands
%     they each comprise. 
%     Similarly for emission_covs, the average of
%     positive-probablility squared-difference matrices is taken to be the
%     initialising setting, with the average constructed similarly.
%     *********************************************************************

    n_trials = length(y);
    [~, x, m] = size(y{1});
    emission_means = zeros(x, m);
    emission_covs  = zeros(x, x, m);
    N = zeros(1, m);
    
    for i = 1:n_trials,
        [means, covs, M] = sequenceEmissionParameters(y{i}, zero_probs{i});
      
        emission_means = bsxfun(@rdivide, bsxfun(@times, emission_means, N) + bsxfun(@times, means, M), N+M);
      
        emission_covs  = bsxfun(@rdivide, bsxfun(@times, emission_covs, reshape(N, [1, size(N)])) ... 
            + bsxfun(@times, covs, reshape(M, [1, size(M)])), reshape(N+M, [1, size(N)]));
        N = N+M;
    end
    
    emission_means = reshape(emission_means, [1, size(emission_means)]);
end