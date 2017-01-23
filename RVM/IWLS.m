function w = IWLS(w, x, y, alpha, max_iters, epsilon)
% Runs IWLS for a max of max_iters iterations, or until
% norm(w_new - w_old) < epsilon

% Inputs:

% w:            (dim_x times C) by 1 vector of weights for each class
%               output. Flattened version of dim_x by C array of weights,
%               with each column holding the weights for one class output.

% x:            N by dim_x array of covariate vectors.

% y:            N by C array of 1-hot target outputs, each having precisely
%               one element with a 1 and C-1 elements with a 0

% alpha:        dim_x by C array of hyperparameters, each for the
%               corresponding element of unflattened w.

% Outputs:

% f:            real number, value of log posterior.

% grad:         (dim_x times C) by 1 grad vector of first order partial
%               differentials wrt each element of w.

% H:            (dim_x times C) by (dim_x times C) Hessian matrix of second
%               order partial differentials wrt each pair of elements of w.

% Author:       Sam Parsons
% Date created: 01/12/2016
% Last amended: 01/12/2016

%     *********************************************************************
%     Check inputs.
%     *********************************************************************

%     All args must be input
    if nargin < 6
        error('all args must be input')
    end
%     w must be a real column  vector
    if ~(isnumeric(w) && isreal(w) && iscolumn(w))
        error('w must be a real column vector')
    end
%     x must be a N by dim_x array of real covariates, with dim_x dividing
%     length(w)
    if ~(isnumeric(x) && isreal(x) && ismatrix(w))
        error('x must be a real matrix')
    end
    l_w = length(w);
    [N, dim_x] = size(x);
    C = l_w / dim_x;
    if C ~= round(C)
        error('size(x, 2) must divide length(w)')
    end
%     y must be an N by C array of 1-hot row vectors
    if ~(islogical(y) && ismatrix(y) && all(size(y) == [N, C]))
        error('y must be an [N, C] array')
    end
    is_1hot = arrayfun(@(row_idx) length(find(y(row_idx, :))) == 1, 1:N);
    if ~all(is_1hot)
        error('all rows of y must be 1-hot')
    end
%     alpha must be [dim_x C] non-negative array
    if ~(isnumeric(alpha) && isreal(alpha) && ismatrix(alpha) &&...
            all(size(alpha) == [dim_x, C]) && all(alpha(:) >= 0))
        error('alpha must be a [dim_x, C] non-negative array')
    end
%     max_iters must be a non-negative integer
    if ~(isnumeric(max_iters) && isreal(max_iters) && isscalar(max_iters) &&...
            (max_iters == round(max_iters)) && (max_iters >= 0))
        error('max_iters must be a non-negative integer')
    end
%     epsilon must be a positive real
    if ~(isnumeric(epsilon) && isreal(epsilon) && isscalar(epsilon) &&...
            (epsilon > 0))
        error('epsilon must be a positive real')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Implements grad and H from weightPosteriorSBLR.m
%     at each iteration and uses them in a naive Newton-Raphson update (no
%     line search). Repeats until convergence, or until max_iters is
%     reached.
%     *********************************************************************

    for iter = 1:max_iters
        fprintf('\tIteration %d\n', iter)
        w_old = w;
        [~, grad, H] = weightPosteriorSBLR(w_old, x, y, alpha);
        w = w_old - H \ grad;
        if norm(w - w_old) < epsilon
            break
        end
    end

end
