function H = posteriorEntropy(p_x_x_next)
% Computes the entropy of the posterior distribution P(X_1:T | Y, U; theta)

% Inputs:

% p_x_x_next:       (m+1) by (m+1) by (T-1) array of joint posteriors
%                   P(X_t:t+1 | Y, U; theta)

% Outputs:

% H:                positive real number giving the entropy of the
%                   posterior

% Author:           Sam Parsons
% Date created:     07/11/2016
% Last amended:     07/11/2016

%     *********************************************************************
%     Check p_x_x_next is valid, i.e. array of non-negative square matrices
%     each summing to 1, with consistent marginals.
%     *********************************************************************
    if nargin < 1
        error('p_x_x_next must be input')
    end
    nd = ndims(p_x_x_next);
    if (nd < 2) || (nd > 3)
        error('p_x_x_next must be [m+1 m+1 T-1] for m>=0 and T>=2')
    end
    [s_p1, s_p2, s_p3] = size(p_x_x_next);
    if s_p1 ~= s_p2
        error('each submatrix of p_x_x_next must be square')
    end
    num_tol = 1e-8;
    is_prob = arrayfun(@(t)...
        abs(1 - sum(sum(p_x_x_next(:, :, t)))) < num_tol, 1:s_p3);
    if ~all(is_prob)
        error('each submatrix of p_x_x_next must be a probability distribution')
    end
    if s_p3 > 1
        is_consistent = arrayfun(@(t)...
            all(abs(sum(p_x_x_next(:, :, t), 2)' - sum(p_x_x_next(:, :, t-1), 1)) < num_tol),...
            2:s_p3);
        if ~is_consistent
            err_msg = ['marginal distribution of each x_t for 2<=t<T must ',...
                'be consistent in both of the matrices it appears in'];
            error(err_msg)
        end
    end
%     *********************************************************************
    
%     *********************************************************************
%     Main body of code. Entropy of P(X_1:2 | Y, U; theta) is calculated
%     unconditionally. Contributions from t=3,...,T are added conditionally
%     using H[P(X_1:t+1 | Y, U; theta)] = H[P(X_1:t | Y, U; theta)] +
%     E[H[P(X_t+1 | X_t, Y, U; theta)]]_X_t, taking the expectation in the
%     second term wrt the marginal posterior P(X_t | Y, U; theta). If T=2
%     then only unconditional extropy described above is calculated. Zero
%     probabilities in entropy calculations are transformed to ones, as
%     0 * log(0) = 0 = 1 * log(1), while matlab thinks 0 * log(0) = NaN
%     *********************************************************************

    p_x12 = p_x_x_next(:, :, 1);
    p_x12(p_x12(:) == 0) = 1;
    H = p_x12(:)' * -log(p_x12(:));
    for t = 2:s_p3
        cond_p = bsxfun(@rdivide, p_x_x_next(:, :, t),...
            sum(p_x_x_next(:, :, t), 2));
        cond_p(isnan(cond_p(:))) = 1;
        cond_p(cond_p(:) == 0) = 1;
        cond_H = arrayfun(@(row_idx)...
            cond_p(row_idx, :) * -log(cond_p(row_idx, :))', 1:s_p2);
        H = H + cond_H * sum(p_x_x_next(:, :, t), 2);
    end

end