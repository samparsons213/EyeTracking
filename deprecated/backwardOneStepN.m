function [p_x, p_x_x_next] = backwardOneStepN(p_x, p_x_next, P, n_timesteps)
% Completes both the backward 'predict' and backward 'update' of one step
% backward in the backward pass, returning both the joint distribution
% p(x_t, x_t+1 | y_1:T, u_1:T) and the marginal p(x_t | y_1:T, u_1:T),
% allowing for non-adjacent data, i.e. the next data point can be more than
% one time step away.

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
% Date created: 01/11/2016
% Last amended: 01/11/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     First 3 arguments must be input
    if nargin < 3
        error('first 3 arguments must be input')
    end
%     p_x must be [1 n+1] probability distribution
    num_tol = 1e-8;
    if ~(isnumeric(p_x) && isreal(p_x) && isrow(p_x) && all(p_x >= 0) &&...
            (abs(1 - sum(p_x)) < num_tol))
        error('p_x must be a [1 n+1] probability matrix')
    end
%     p_x_next must be a [1 n+1] probability distribution
    dim_x = length(p_x);
    if ~(isnumeric(p_x_next) && isreal(p_x_next) && isrow(p_x_next) &&...
            (length(p_x_next) == dim_x) && all(p_x_next >= 0) &&...
            (abs(1 - sum(p_x_next)) < num_tol))
        error('p_x_next must be a [1 n+1] probability matrix')
    end
%     P must be a [n+1 n+1] probability distribution (all rows sum to 1).
%     Probability condition checked in forwardBackward.m
    if ~(isnumeric(P) && isreal(P) && ismatrix(P) && all(size(P) == dim_x))
        error('P must be a real square matrix of size length(p_x)')
    end
%     n_timesteps must be a positive integer, if entered. If not entered
%     then it is set to its default value of 1
    if nargin < 4
        n_timesteps = 1;
    elseif ~(isnumeric(n_timesteps) && isreal(n_timesteps) &&...
            isscalar(n_timesteps) && (n_timesteps == round(n_timesteps)) &&...
            (n_timesteps > 0))
        error('n_timesteps must be a positive integer')
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

    p_x_x_next = bsxfun(@times, p_x', (P^n_timesteps));
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