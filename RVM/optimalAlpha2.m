function [alpha, w, relevant_dims] = optimalAlpha2(x, y, epsilon, max_iters, eff_inf, alpha)
% Returns the type 2 mle of alpha given the feature vectors x and the data
% y, with an optional initial value for alpha. Estimate is considered to
% have converged when largest change in any of its non-inf elements' (where
% non-inf refers to elements with values less than eff_inf) values is less
% than epsilon, or after max_iters iterations have been completed.

% Adjusted so one alpha is prior to all w for correpsonding dimension of x

% Inputs:

% x:            N by D array of D dimensional feature vectors

% y:            N by C logical array with exactly 1 true value in each row.

% epsilon:      positive real giving convergence criteria for alpha
%               estimate

% max_iters:    positive integer giving maximum number of iterations that
%               must be completed

% eff_inf:      positive real number giving the value of any individual
%               alpha considered to constitute effective infinity, i.e.
%               corresponding dimension of x is trimmed from that class's
%               classifier (optional, default value is 1000)

% alpha:        D by 1 array of positive reals giving initial estimate of
%               alpha (optional)

% Outputs:

% alpha:        D by 1 array of positive reals giving final estimate of
%               alpha

% w:            value of w at mode of posterior with final value of alpha
%               as hyperparameter

% Author:       Sam Parsons
% Date created: 07/12/2016
% Last amended: 07/12/2016

%     *********************************************************************
%     Validate inputs.
%     *********************************************************************

%     First 4 args must be input
    if nargin < 4
        error('first 4 args must be input')
    end
%     x must be a N by D array of real covariates
    if ~(isnumeric(x) && isreal(x) && ismatrix(x))
        error('x must be a real matrix')
    end
    [N, D] = size(x);
%     y must be an N by C array of 1-hot row vectors
    if ~(islogical(y) && ismatrix(y) && (size(y, 1) == N))
        error('y must be an [size(x, 1), C] logical array for some C')
    end
    C = size(y, 2);
    is_1hot = arrayfun(@(row_idx) length(find(y(row_idx, :))) == 1, 1:N);
    if ~all(is_1hot)
        error('all rows of y must be 1-hot')
    end
%     epsilon must be a positive real
    if ~(isnumeric(epsilon) && isreal(epsilon) && isscalar(epsilon) &&...
            (epsilon > 0))
        error('epsilon must be a positive real')
    end
%     max_iters must be a positive integer
    if ~(isnumeric(max_iters) && isreal(max_iters) && isscalar(max_iters) &&...
            (max_iters == round(max_iters)) && (max_iters >= 1))
        error('max_iters must be a positive integer')
    end
%     If input, eff_inf must be a positive real
    if nargin > 4
        if ~(isnumeric(eff_inf) && isreal(eff_inf) && isscalar(eff_inf) &&...
                (eff_inf > 0))
            error('eff_inf must be a positive real')
        end
    else
        eff_inf = 1000;
    end
%     If input, alpha must be a [D, 1] non-negative array
    if nargin > 5
        if ~(isnumeric(alpha) && isreal(alpha) && iscolumn(alpha) &&...
                (length(alpha) == D) && all(alpha(:) >= 0))
            error('alpha must be a [size(x, 1), size(y, 2)] non-negative array')
        end
    else
        alpha = randn(D, 1) .^ 2;
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Iteratively optimises alpha until convergence
%     criteria is reached. Alpha is optimised as per Bishop book on RVMs
%     adapted to multi class classification. Laplace approximation to
%     (unnormalised) posterior of weights (with sparsity prior given by
%     current value of alpha) is used to estimate the normalisation
%     constant of the true posterior (i.e. the likelihood of the data given
%     alpha). This is maximised and used to make a new Laplace
%     approximation, and this 2 stage procedure is repeated until
%     convergence.
%     *********************************************************************

    opts.Display = 'off';
    opts.Method = 'newton';
    opts.optTol = 1e-8;
    w = rand(D, C);
    dp = x * w;
    dp = bsxfun(@minus, dp, max(dp, [], 2));
    probs = bsxfun(@rdivide, exp(dp), sum(exp(dp), 2));
    l = sum(log(probs(y))) + sum(log(alpha(:)))/2 - sum(sum(alpha.*(w.^2)))/2;
    for iter = 1:max_iters
%         alpha_old = alpha;
        l_old = l;
        fprintf('Iteration %d\n', iter)
        fprintf('\tFinding Laplace approximation to posterior\n')
        alpha_rep = repmat(alpha, 1, C);
        w = minFunc(@(theta) weightPosteriorSBLR(theta, x, y, alpha_rep, true),...
            w(:), opts);
        fprintf('\tOptimising alpha\n')
        [~, ~, H] = weightPosteriorSBLR(w, x, y, alpha_rep, false);
        Sigma = -inv(H);
        alpha = 1 ./ sum(reshape(w .^ 2 + diag(Sigma), D, C), 2);
        relevant_dims = alpha < eff_inf;
%         delta = max(abs(alpha(relevant_dims) - alpha_old(relevant_dims)));
        w = reshape(w, D, C);
        w(~relevant_dims) = 0;
        dp = x * w;
        dp = bsxfun(@minus, dp, max(dp, [], 2));
        probs = bsxfun(@rdivide, exp(dp), sum(exp(dp), 2));
        l = sum(log(probs(y))) + sum(log(alpha(:)))/2 - sum(sum(alpha.*(w.^2)))/2;
        delta = l - l_old;
        fprintf('\t max abs delta = %f\n', delta)
        if abs(delta) < epsilon
            break
        end
    end
%     w = reshape(w, D, C);
    w(~relevant_dims, :) = 0;

end