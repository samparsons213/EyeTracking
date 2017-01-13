function [l, grad] = ETTL(p_x, p_x_x_next, theta, y)
% Computes the expected total log likelihood and its grad vector

% Inputs:

% p_x:              T by k array of posterior distributions
%                   p(x_t | Y, U; theta)

% p_x_x_next:       k by k by T-1 array of posterior distributions
%                   p(x_t, x_t+1 | Y, U; theta)

% theta:            k +k^2 + 2*k + 3*k by 1 parameter vector. When
%                   reconstructed, parameters should be:

%       pi_1:             1 by k prior distribution for x_1

%       P:                k by k prior transition matrix

%       emission_means:   1 by 2 by k array of mean vectors for each latent
%                         state

%       emission_covs:    2 by 2 by k array of covariance matrices for each
%                         latent state

% y:                T by 2 by k array of difference vectors, transformed
%                   according to each latent state

% Outputs:

% l:                real number giving the expected total log likelihood of
%                   X and Y for one person

% grad:             n_parms by 1 grad vector, order is pi_1(:), P(:),
%                   emission_means(:), ~emission_covs(:) (for each matrix
%                   in emission_covs, only elements 1,2,4 are in grad
%                   vector, as element 3 = element 2)

% Author:           Sam Parsons
% Date created:     04/11/16
% Last amended:     04/11/16

%     *********************************************************************
%     Main body of code. Function val computed as efficiently as possible.
%     Grad vector computed via gradETTL.m function.
%     *********************************************************************

    num_tol = 1e-6;
    k = size(p_x, 2);
    pi_1 = reshape(theta(1:k), 1, k);
    pi_1 = max(num_tol, pi_1);
    P_idx = (k+1):(k+k^2);
    P = reshape(theta(P_idx), k, k);
    P = max(num_tol, P);
    em_idx = (k+k^2+1):(k+k^2+2*k);
    emission_means = reshape(theta(em_idx), 1, 2, k);
    ec_idx1 = (k+k^2+2*k+1):(k+k^2+2*k+3*k);
    s_ec = [2, 2, k];
    nel = prod(s_ec);
    ec_idx2 = sort([1:4:nel, 2:4:nel, 4:4:nel]);
    emission_covs = zeros(s_ec);
    emission_covs(ec_idx2) = theta(ec_idx1);
    ec_idx3 = 3:4:nel;
    emission_covs(ec_idx3) = emission_covs(ec_idx3-1);
    
    l_x_1 = p_x(1, :) * log(pi_1)';
    l_x_2T_temp = sum(p_x_x_next, 3);
    l_x_2T = l_x_2T_temp(:)' * log(P(:));
    
    k = length(pi_1);
    log_dets = arrayfun(@(xidx)...
        log(det(emission_covs(:, :, xidx))), 1:k);
    l_f_temp1 = bsxfun(@minus, y, emission_means);
    l_f_temp2 = cell2mat(arrayfun(@(xidx)...
        sum((l_f_temp1(:, :, xidx) / emission_covs(:, :, xidx)) .* l_f_temp1(:, :, xidx), 2),...
        1:k, 'UniformOutput', false));
    l_f_temp3 = -bsxfun(@plus, l_f_temp2, log_dets) ./ 2;
    l_f = sum(sum(l_f_temp3 .* p_x));
    
    l = l_x_1 + l_x_2T + l_f;
    
    grad = gradETLL(p_x, p_x_x_next, pi_1, P, emission_means,...
        emission_covs, y);

end