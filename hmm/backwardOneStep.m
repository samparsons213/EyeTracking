function [p_x, p_x_x_next] = backwardOneStep(p_x, p_x_next, P)
% Completes both the backward 'predict' and backward 'update' of one step
% backward in the backward pass, returning both the joint distribution
% p(x_t, x_t+1 | y_1:T, u_1:T) and the marginal p(x_t | y_1:T, u_1:T)

% Inputs:

% p_x:          1 by (n+1) probability distribution (sums to 1), giving the
%               filtered distribution p(x_t | y_1:t, u_1:t)

% p_x_next:     1 by (n+1) probability distribution (sums to 1), giving the
%               smoothed distribution p(x_t+1 | y_1:T, u_1:T)

% P:            (n+1) by (n+1) matrix of transition probabilties, with each 
%               row summing to 1

% Outputs:

% p_x:          1 by (n+1) probability distribution (sums to 1), giving the
%               smoothed distribution p(x_t | y_1:T, u_1:T)

% p_x_x_next:   (n+1) by (n+1) probability matrix (sums to 1), giving the
%               joint probabilities p(x_t, x_t+1 | y_1:T, u_1:T)

% Author:       Sam Parsons
% Date created: 21/09/2016
% Last amended: 21/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

    isMatrix = @(x) isnumeric(x) && isreal(x) && ismatrix(x);
    isPositiveMatrix = @(x) isMatrix(x) && all(x(:) >= 0);
    
    if nargin < 3
        error('all 3 arguments must be input')
    end

    num_tol = 1e-8;
    if ~(isPositiveMatrix(p_x) && (abs(1 - sum(p_x)) < num_tol))
        error('p_x must be a [1 n+1] probability matrix')
    end
    

    dim_x = length(p_x);
    if ~(isPositiveMatrix(p_x_next) && (length(p_x_next) == dim_x) && ...
            (abs(1 - sum(p_x_next)) < num_tol))
        error('p_x_next must be a [1 n+1] probability matrix')
    end

    if ~(isMatrix(P) && all(size(P) == dim_x))
        error('P must be a real square matrix of size length(p_x)')
    end
    
%     *********************************************************************

%     *********************************************************************
%     Main body of function. p_x_x_next is computed first, in two stages.
%     First stage is to add conditioning of x on x_next to get
%     p(x_t | y_1:t, u_1:t, x_t+1). Second stage is to multiply the output
%     of first stage by p(x_t+1 | y_1:T, u_1:T). p_x is computed by
%     marginalising from p_x_x_next, i.e. by summing rows of p_x_x_next and
%     transposing
%     *********************************************************************

    p_x_x_next = bsxfun(@times, p_x', P);
%     It can sometimes be that all entries in one or more columns of
%     p_x_x_next = p(x_t, x_t+1 | y_1:t, u_1:t) are zero. This will happen
%     if, for example, some elements of p_x = p(x_t | y_1:t, u_1:t) are
%     zero, and the complementary rows of the columns in question in P are
%     also 0. The natural interpretation of this is that the prediction
%     step of the forward step places 0 probability on those states at time
%     t+1; this makes conditioning on the x_t+1 value for those colunms ill
%     defined, and all entries in those columns of the conditional
%     probability matrix p_x_x_next = p(x_t | y_1:t, u_1:t, x_t+1)
%     calculated next will be NaN. The solution proposed here is to replace
%     those NaNs with zeros (achieved by replacing zero elements of
%     sum(p_x_x_next, 1) with ones). This may result in p_x summing to less
%     than 1, though the difference should be negligable as smoothed
%     distribution p(x_t+1 | y_1:T, u_1:T) should also place zero
%     probability on these entries. Renormalising after p_x =
%     p(x_t | y_1:T, u_1:T) should produce the correct distribution, if not
%     then all entries of p(x_t | y_1:t, u_1:t, x_t+1) are NaN and the
%     procedure should return an error.
    p_x_next_filter = sum(p_x_x_next, 1);
    p_x_next_filter(p_x_next_filter == 0) = 1;
    p_x_x_next = bsxfun(@rdivide, p_x_x_next, p_x_next_filter);
    p_x_x_next = bsxfun(@times, p_x_x_next, p_x_next);
    p_x = sum(p_x_x_next, 2)';
    s_p_x = sum(p_x);
    if s_p_x == 0
        error('p_x contains all zeros')
    end
    p_x = p_x ./ s_p_x;

end