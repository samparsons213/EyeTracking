function p_x = priorMarginals(pi_1, P, T)
% Returns the prior marginals p(x_t | theta) at each time t implied by the
% distribution pi_1 = p(x_1) and the transition matrix P:
% P(i, j) = p(x_t+1 | x_t), for times 1 to T

% Inputs:

% pi_1:             1 by dim_x probablity distribution (sums to 1)

% P:                dim_x by dim_x transition matrix (each row sums to 1)

% T:                positive integer

% Outputs:

% p_x:              T by dim_x array of probability distributions p(x_t)
%                   (each row sums to 1)

% Author:           Sam Parsons
% Date created:     30/09/2016
% Last amended:     30/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

%     All  arguments must be input
    if nargin < 3
        error('all 3 arguments must be input')
    end
%     pi_1 must be a [1 dim_x] probability vector
    num_tol = 1e-8;
    if ~(isnumeric(pi_1) && isreal(pi_1) && isrow(pi_1) && all(pi_1 >= 0) &&...
            (abs(1 - sum(pi_1)) < num_tol))
        error('pi_1 must be a [1 dim_x] probability vector')
    end
    dim_x = length(pi_1);
%     P must be a [dim_x dim_x] matrix, with each row a probability vector
    if ~(isnumeric(P) && isreal(P) && ismatrix(P) && all(size(P) == dim_x))
        error('P must be a [dim_x dim_x] matrix')
    end
    is_prob_dist = arrayfun(@(x_idx) abs(1 - sum(P(x_idx, :))) < num_tol,...
        1:dim_x);
    if ~(all(P(:) >= 0) && all(is_prob_dist))
        error('each row of P must be a probability distribution')
    end
%     T must be a positive integer
    if ~(isnumeric(T) && isreal(T) && (T == round(T)) && (T > 0))
        error('T must be a positive integer')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. p_x has p_x(1, :) = pi_1, and each subsequent row
%     is just the row above propagated forward by P
%     *********************************************************************

    p_x = [pi_1; zeros(T-1, dim_x)];
    for t = 2:T
        p_x(t, :) = p_x(t-1, :) * P;
    end

end