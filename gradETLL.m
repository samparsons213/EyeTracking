function grad = gradETLL(p_x, p_x_x_next, pi_1, P, emission_means,...
    emission_covs, y)
% Returns the grad vector of the expected total log likelihood wrt the
% model parameters.

% Inputs:

% p_x:              T by k array of posterior distributions
%                   p(x_t | Y, U; theta)

% p_x_x_next:       k by k by T-1 array of posterior distributions
%                   p(x_t, x_t+1 | Y, U; theta)

% pi_1:             1 by k prior distribution for x_1

% P:                k by k prior transition matrix

% emission_means:   1 by 2 by k array of mean vectors for each latent state

% emission_covs:    2 by 2 by k array of covariance matrices for each
%                   latent state

% y:                T by 2 by k array of difference vectors, transformed
%                   according to each latent state

% Outputs:

% grad:             n_parms by 1 grad vector, order is pi_1(:), P(:),
%                   emission_means(:), ~emission_covs(:) (for each matrix
%                   in emission_covs, only elements 1,2,4 are in grad
%                   vector, as element 3 = element 2)

% Author:           Sam Parsons
% Date created:     04/11/16
% Last amended:     04/11/16

%     *********************************************************************
%     Main body of code. Grad vector computed as efficiently as possible.
%     As diff wrt emission_covs(i, j, k) needs diff Sigma^-1_ij wrt
%     Sigma_rs for all i,j,r,s, we compute this as a necesary first step.
%     *********************************************************************

    num_tol = 1e-6;
    grad_pi_1 = p_x(1, :) ./ max(num_tol, pi_1);
    grad_P = sum(p_x_x_next, 3) ./ max(num_tol, P);
    k = length(pi_1);
    T = size(y, 1);
    grad_em_temp1 = reshape(p_x, T, 1, k);
    grad_em_temp2 = bsxfun(@minus, y, emission_means);
    grad_em_temp3 = cell2mat(arrayfun(@(xidx)...
        grad_em_temp2(:, :, xidx) / emission_covs(:, :, xidx),...
        reshape(1:k, 1, 1, k), 'UniformOutput', false));
    grad_em = sum(bsxfun(@times, grad_em_temp1, grad_em_temp3));
    
%     dimension order for dSig^-1_ij / dSig_rs is i, j, r, s. Off diagonal
%     elements are treated first, then diagonal. For off diagonal, only
%     r > s are considered
    d = size(emission_covs, 1);
    grad_ec_temp2 = grad_em_temp2;
    grad_ec = zeros(d, d, k);
    for xidx = 1:k
        grad_ec_temp1 = reshape(p_x(:, xidx), 1, 1, T);
        Sigma = emission_covs(:, :, xidx);
        dSigIdSig = zeros(d, d, d, d);
        for r = 1:d
            for s = 1:(r-1)
                alpha = zeros(d);
                alpha(r, s) = 1;
                alpha(s, r) = 1;
                dSigIdSig(:, :, r, s) = -(Sigma \ alpha) / Sigma;
            end
            alpha = zeros(d);
            alpha(r, r) = 1;
            dSigIdSig(:, :, r, r) = -(Sigma \ alpha) / Sigma;
        end
        grad_ec_temp3 = cell2mat(arrayfun(@(t)...
            grad_ec_temp2(t, :, xidx)' * grad_ec_temp2(t, :, xidx),...
            reshape(1:T, 1, 1, T), 'UniformOutput', false));
        grad_ec_temp4 =...
            sum(bsxfun(@times, grad_ec_temp1, grad_ec_temp3), 3);
        for idx1 = 1:d
            for idx2 = 1:idx1
                grad_ec_temp5 =...
                    grad_ec_temp4 .* dSigIdSig(:, :, idx1, idx2);
                grad_ec(idx1, idx2, xidx) = sum(grad_ec_temp5(:));
            end
        end
        grad_ec(:, :, xidx) = -(grad_ec(:, :, xidx) +...
            sum(p_x(:, xidx)) .* inv(Sigma)) ./ 2;
    end
    nel = numel(emission_covs);
    grad_ec_idx = sort([1:4:nel, 2:4:nel, 4:4:nel])';
    
    grad = [grad_pi_1'; grad_P(:); grad_em(:); grad_ec(grad_ec_idx)];

end