function [pi_1, P, emission_means, emission_covs] =...
    generateRandomParameters(dim_x, alpha)
% Returns randomly initialised parameters for the proposed HMM, taking one
% hyperparameter as input for producing random probability distributions.
% Random covariance matrices are generated as Wishart_2(2, I), thereby
% having E[emission_covs(:, :, i)] = 2I for all i

% Inputs:

% dim_x:            positive integer at least 2, giving the number of
%                   latent states, those being 'no movement' and each of m
%                   canonical latent directions

% alpha:            positive real number giving the single Dirichlet
%                   parameter for all randomly drawn distributions, i.e. if
%                   p is a randomly drawn 1 by n distribution it is sampled
%                   from D(alpha*ones(1, n))

% Outputs:

% pi_1:             1 by (m+1) latent prior for x_1

% P:                (m+1) by (m+1) latent transition matrix, with row a
%                   probability distribution

% emission_means:   1 by 2 by (m+1) array of emission mean vectors

% emission_covs:    2 by 2 by (m+1) array of emission covariance matrices

% Author:           Sam Parsons
% Date created:     23/09/2016
% Last amended:     23/09/2016

%     *********************************************************************
%     Check inputs
%     *********************************************************************

%     dim_x must be input
    if nargin < 1
        error('dim_x must be input')
    end
%     dim_x must be a positive integer at least 2
    if ~(isnumeric(dim_x) && isreal(dim_x) && isscalar(dim_x) &&...
            (dim_x == round(dim_x)) && (dim_x >= 2))
        error('dim_x must be a positive integer at least 2')
    end
%     if input, alpha must be a positive real, if not input it is set to 1
    if nargin == 2
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
%     emission_covs is drawn Wishart_2(2, V) as described above.
%     *********************************************************************

    random_gammas = gamrnd(alpha, 1, (dim_x+1), dim_x);
    pi_1 = random_gammas(1, :) ./ sum(random_gammas(1, :));
    P = bsxfun(@rdivide, random_gammas(2:end, :),...
        sum(random_gammas(2:end, :), 2));
    emission_means = randn(1, 2, dim_x);
    emission_covs = randn(2, 2, dim_x);
    emission_covs = cell2mat(arrayfun(@(x_idx)...
        emission_covs(:, :, x_idx) * emission_covs(:, :, x_idx)',...
        reshape(1:dim_x, 1, 1, dim_x), 'UniformOutput', false));

end