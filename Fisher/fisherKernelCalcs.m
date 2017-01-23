function [gram_matrix, distances] = fisherKernelCalcs(fisher_fv_ctrl,...
    fisher_fv_ad, fisher_fv_pca, fisher_info, n_exp_sum)
% Calculates the Gram matrix of approximate inner products, summing over
% experiments, and the distance

% Inputs:

% fisher_fv_ctrl:       n_parms by n_ctrl by n_experiments array of score
%                       vectors evaluated at the fitted parameters for each
%                       experiment, taking each control individual's data
%                       for that experiment as the data for the
%                       log-likelihood

% fisher_fv_ad:         n_parms by n_ad by n_experiments array of score
%                       vectors evaluated at the fitted parameters for each
%                       experiment, taking each AD individual's data for
%                       that experiment as the data for the log-likelihood

% fisher_fv_pca:        n_parms by n_pca by n_experiments array of score
%                       vectors evaluated at the fitted parameters for each
%                       experiment, taking each PCA individual's data for
%                       that experiment as the data for the log-likelihood

% fisher_info:          n_parms by n_parms by n_experiments array of
%                       approximate fisher information matrices (at least
%                       the first n_exp_sum of which have actually been
%                       computed)

% n_exp_sum:            positive integer giving the number of experiments
%                       to be included in the computation of inner products
%                       and distances

% Outputs:

% gram_matrix:          n_patients by n_patients array of fisher kernel
%                       inner products

% distances:            n_patients by n_patients array of fisher kernel
%                       distances

% Author:               Sam Parsons
% Date created:         18/11/2016
% Last amended:         18/11/2016

%     *********************************************************************
%     Validate input arguments.
%     *********************************************************************

%     All args must be input
    if nargin < 5
        error('All errors must be input')
    end
    
%     fisher_fv_ctrl must be [n_parms n_ctrl n_experiments] real array for
%     n>=1
    if ~(isnumeric(fisher_fv_ctrl) && isreal(fisher_fv_ctrl) &&...
            (ndims(fisher_fv_ctrl) > 1) && (ndims(fisher_fv_ctrl) < 4))
        error('fisher_fv_ctrl must be a [n_parms n_ctrl n_experiments] real array')
    end
    [n_parms, n_ctrl, n_experiments] = size(fisher_fv_ctrl);
    
%     fisher_fv_ad must be [n_parms n_ad n_experiments] real array for
%     n>=1
    if ~(isnumeric(fisher_fv_ad) && isreal(fisher_fv_ad) &&...
            (ndims(fisher_fv_ad) > 1) && (ndims(fisher_fv_ad) < 4) &&...
            (size(fisher_fv_ad, 1) == n_parms) &&...
            (size(fisher_fv_ad, 3) == n_experiments))
        error('fisher_fv_ad must be a [n_parms n_ad n_experiments] real array')
    end
    n_ad = size(fisher_fv_ad, 2);
    
%     fisher_fv_pca must be [n_parms n_pca n_experiments] real array for
%     n>=1
    if ~(isnumeric(fisher_fv_pca) && isreal(fisher_fv_pca) &&...
            (ndims(fisher_fv_pca) > 1) && (ndims(fisher_fv_pca) < 4) &&...
            (size(fisher_fv_pca, 1) == n_parms) &&...
            (size(fisher_fv_pca, 3) == n_experiments))
        error('fisher_fv_pca must be a [n_parms n_pca n_experiments] real array')
    end
    n_pca = size(fisher_fv_pca, 2);
    
%     fisher_info must be [n_parms n_parms n_experiments] array of
%     real symmetric psd matrices
    [s_fi1, s_fi2, s_fi3] = size(fisher_info);
    s_fi = [s_fi1, s_fi2, s_fi3];
    num_tol = 1e-10;
    is_spsd = arrayfun(@(exp_idx) issymmetric(fisher_info(:, :, exp_idx)) &&...
        min(eig(fisher_info(:, :, exp_idx))) > -num_tol, 1:n_experiments);
    if ~(isnumeric(fisher_info) && isreal(fisher_info) &&...
            all(s_fi == [n_parms, n_parms, n_experiments]) && all(is_spsd))
        error('fisher_info must be [n_parms n_parms n_experiments] array of real symmetric psd matrices')
    end
    
%     n_exp_sum must be a positive integer between 1 and n_experiments
    if ~(isnumeric(n_exp_sum) && isreal(n_exp_sum) && isscalar(n_exp_sum) &&...
            (n_exp_sum == round(n_exp_sum)) && (n_exp_sum >= 1) &&...
            (n_exp_sum <= n_experiments))
        error('n_exp_sum must be a positive integer between 1 and n_experiments')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. Inner products between two patients are defined as
%     the sum of the fk inner products between their fisher feature vectors
%     in each of the experiments in 1 to n_exp_sum. Distances are similarly
%     as the sum of the distances (square roots of auto inner products of
%     difference vectors) between their fisher feature vectors in each of
%     the experiments in 1 to n_exp_sum.
%     *********************************************************************

% %     fisher_info_inv = cell2mat(arrayfun(@(exp_idx) inv(fisher_info(:, :, exp_idx)),...
% %         reshape(1:n_exp_sum, 1, 1, n_exp_sum), 'UniformOutput', false));
    fisher_info_inv = cell2mat(arrayfun(@(exp_idx) pinv(fisher_info(:, :, exp_idx)),...
        reshape(1:n_exp_sum, 1, 1, n_exp_sum), 'UniformOutput', false));
    fisher_fv = cat(2, fisher_fv_ctrl(:, :, 1:n_exp_sum), fisher_fv_ad(:, :, 1:n_exp_sum),...
        fisher_fv_pca(:, :, 1:n_exp_sum));
% %     gram_matrix = cell2mat(arrayfun(@(exp_idx) fisher_fv(:, :, exp_idx)' *...
% %         fisher_info_inv(:, :, exp_idx) * fisher_fv(:, :, exp_idx),...
% %         reshape(1:n_exp_sum, 1, 1, n_exp_sum), 'UniformOutput', false));
    n_patients = n_ctrl + n_ad + n_pca;
    gram_matrix = cell2mat(arrayfun(@(exp_idx) cell2mat(arrayfun(@(idx1) arrayfun(@(idx2)...
        fisher_fv(:, idx1, exp_idx)' * fisher_info_inv(:, :, exp_idx) *...
        fisher_fv(:, idx2, exp_idx), 1:n_patients), (1:n_patients)',...
        'UniformOutput', false)), reshape(1:n_exp_sum, 1, 1, n_exp_sum),...
        'UniformOutput', false));
    gram_matrix = cell2mat(arrayfun(@(exp_idx) (gram_matrix(:, :, exp_idx) +...
        gram_matrix(:, :, exp_idx)') ./ 2,...
        reshape(1:n_exp_sum, 1, 1, n_exp_sum), 'UniformOutput', false));
    
    n_patients = n_ctrl + n_ad + n_pca;
    distances = cell2mat(arrayfun(@(exp_idx) cell2mat(arrayfun(@(idx1)...
        arrayfun(@(idx2)...
        sqrt(max(num_tol, (fisher_fv(:, idx1, exp_idx) - fisher_fv(:, idx2, exp_idx))' *...
        fisher_info_inv(:, :, exp_idx) *...
        (fisher_fv(:, idx1, exp_idx) - fisher_fv(:, idx2, exp_idx)))),...
        1:n_patients), (1:n_patients)', 'UniformOutput', false)),...
        reshape(1:n_exp_sum, 1, 1, n_exp_sum), 'UniformOutput', false));
%     distances = sum(distances, 3);

end