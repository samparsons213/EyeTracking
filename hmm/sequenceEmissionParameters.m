function [emission_means, emission_covs, N] = sequenceEmissionParameters(y, zero_probs)
% Returns sample mean vectors and covariance matrices of y (transformed
% difference data of eye tracks) of all observations not given zero
% probablity (for having y(t, 1, x_idx) = log(0) or log(-z) for some z>0
% for that t) for each latent state

% Inputs:

% y:                n by 2 by (m+1) array of transformed eye tracking
%                   difference data. First element of third dim is
%                   untransformed, corresponding to 'no movement' latent
%                   state, remaining elements are transformed according to
%                   corresponding columns of l_dirs

% zero_probs:       n by m binary array, where zeros correspond to the
%                   latent directions that have zero posterior probability,
%                   and ones to all other latent directions

% Outputs:

% emission_means:   1 by 2 by (m+1) array of mean vectors for the Gaussian 
%                   emission densities for each latent state

% emission_covs:    2 by 2 by (m+1) array of covariance matrices for
%                   the Gaussian emission densities for each latent state

% N                 number of rows this estimate is based on

% Author:           Sam Parsons
% Date created:     26/09/2016
% Last amended:     26/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

    is3DTensor = @(x) isnumeric(x) && isreal(x) && (ndims(x) == 3);
    isBinaryMatrix = @(x) isnumeric(x) && all((x(:) == 0) | x(:) == 1);
    
    if nargin < 2
        error('all 2 arguments must be input')
    end

    s_y = size(y);
    if ~(is3DTensor(y) && (s_y(2) == 2))
        error('y must be a [n 2 m+1] real array')
    end

    if ~(isBinaryMatrix(zero_probs) &&...
            all(size(zero_probs) == [s_y(1), s_y(3)-1]))
        error('zero_probs must be a [size(y, 1) size(y, 3)-1] binary array')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. 
%     emission_means takes the average along dim 1 of y,
%     subject to zero_probs = 1 for that row and state combination
%     (zero_probs is augmented with ones for state 1 as the no movement
%     state always has positive probablity). 
%     Similarly for emission_covs,
%     the average of positive-probablility squared-difference matrices is
%     taken to be the initialising setting.
%     *********************************************************************

    zero_probs = reshape([ones(s_y(1), 1), zero_probs], s_y(1), 1, s_y(3));
    emission_means = bsxfun(@rdivide, sum(bsxfun(@times, y, zero_probs), 1), sum(zero_probs, 1));
    
    diffs = bsxfun(@minus, y, emission_means);
    emission_covs = zeros(s_y(2), s_y(2), s_y(3));
    for i = 1:size(diffs, 3),
        diff_vec = squeeze(diffs(:, :, i));
        Z = squeeze(zero_probs(:,:,i));
        emission_covs(:, :, i) = (diff_vec' * bsxfun(@times, diff_vec, Z))/sum(Z);
    end
    
    N = squeeze(sum(zero_probs, 1))';
    emission_means = squeeze(emission_means);
end