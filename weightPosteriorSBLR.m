function [f, grad, H] = weightPosteriorSBLR(w, x, y, alpha, negate)
% Evaluates the log posterior of the weights given the data and the
% hyperparameters, up to the log normalising constant, for the purposes of
% type 2 mle of hyperparameters for sparse Bayes logistic regression with
% C>2 classes.

% Inputs:

% w:            (dim_x times C) by 1 vector of weights for each class
%               output. Flattened version of dim_x by C array of weights,
%               with each column holding the weights for one class output.

% x:            N by dim_x array of covariate vectors.

% y:            N by C array of 1-hot target outputs, each having precisely
%               one element with a 1 and C-1 elements with a 0

% alpha:        dim_x by C array of hyperparameters, each for the
%               corresponding element of unflattened w.

% negate:       logical scalar, false corresponds to evaluating log
%               posterior, true corresponds to evaluating negative log
%               posterior (use for eg minFunc)

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
    if nargin < 5
        error('all args must be input')
    end
%     w must be a real column  vector
    if ~(isnumeric(w) && isreal(w) && iscolumn(w))
        error('w must be a real column vector')
    end
%     x must be a N by dim_x array of real covariates, with dim_x dividing
%     length(w)
    if ~(isnumeric(x) && isreal(x) && ismatrix(x))
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
%     negate must be a logical scalar
    if ~(islogical(negate) && isscalar(negate))
        error('negate must be a logical scalar')
    end
    
%     *********************************************************************
%     Main body of code. Evaluate function value by evaluating normaliser
%     of softmax for each observation, and subtracting the sum over obs
%     from the sum of true dot products for each obs. From this subtract
%     the sum over classes of half w_j'*A_j*w_j.

%     Reshape w to its correct form.
    w = reshape(w, dim_x, C);
%     Compute all dot products.
    dot_products = x * w;
%     Shift by max on each row, record row maximums.
    max_dp = max(dot_products, [], 2);
    shifted_dp = bsxfun(@minus, dot_products, max_dp);
%     Exponentiate.
    exp_sh_dp = exp(shifted_dp);
%     Sum true dot products.
    s_dp = sum(dot_products(y));
%     Compute sum of quadratic forms.
    Aw = alpha .* w;
    s_q_forms = sum(sum(w .* Aw));
%     Compute f.
    f = s_dp - (sum(max_dp) + sum(log(sum(exp_sh_dp, 2)))) - s_q_forms/2;
    
%     grad is evaluated by computing matrix product of x' with (y-softmax)
%     and subtracting Aw, and then flattening
    smax = bsxfun(@rdivide, exp_sh_dp, sum(exp_sh_dp, 2));
    grad = x' * (y - smax) - Aw;
    grad = grad(:);
    
%     Negate f and grad if negate set to true
    if negate
        f = -f;
        grad = -grad;
    end
    
%     If H is requested, it is computed as per my notes.
    if nargout == 3
        dim_x_sq = dim_x ^ 2;
        diag_idx = (1:(dim_x+1):dim_x_sq)';
        sigmoid_diff = smax .* (1-smax);
        H = zeros(dim_x * C);
        for idx1 = 1:C
            H_idx1 = ((idx1-1) * dim_x + 1):(idx1 * dim_x);
            x_temp = bsxfun(@times, x, sigmoid_diff(:, idx1));
            H_temp = -(x_temp' * x);
            H_temp(diag_idx) = H_temp(diag_idx) - alpha(:, idx1);
            H(H_idx1, H_idx1) = H_temp;
            for idx2 = (idx1+1):C
                H_idx2 = ((idx2-1) * dim_x + 1):(idx2 * dim_x);
                x_temp = bsxfun(@times, x, smax(:, idx1) .* smax(:, idx2));
                H(H_idx1, H_idx2) = x_temp' * x;
                H(H_idx2, H_idx1) = H(H_idx1, H_idx2);
            end
        end
        if negate
            H = -H;
        end
    end

end