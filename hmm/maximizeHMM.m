function [pi_1, P, emission_means, emission_covs] = maximizeHMM(y, p_x, p_x_x_next, min_eig)
% Returns the parameters that maximise the expected total log-likelihood,
% maximisation step of EM, for many people's data

% Inputs:

% y:                n_people by 1 cell, each element of which is an n by 2
%                   by (m+1) array of transformed eye tracking difference
%                   data. First element of third dim is untransformed,
%                   corresponding to 'no movement' latent state, remaining
%                   elements are transformed according to corresponding
%                   columns of l_dirs

% p_x:              n_people by cell, ecah element of which is an n_p by
%                   (m+1) array of probability distributions (each row sums
%                   to 1), giving smoothed distributions
%                   p(x_p,t | y_p,1:T, u_p,1:T)

% p_x_x_next:       n_people by 1 cell, each element of which is an (m+1)
%                   by (m+1) by (n_p-1) array of probability distributions
%                   (each matrix sums to 1), giving smoothed distributions
%                   p(x_p,t, x_p,t+1 | y_p,1:T, u_p,1:T)

% min_eig:          positive real number giving the minimum acceptable
%                   eigenvalue  of estimated covariance matrices

% Outputs:

% pi_1:             1 by (m+1) probability distribution (sums to 1) giving
%                   the prior distribution of x_1

% P:                (n+1) by (n+1) matrix of transition probabilties, with
%                   each row summing to 1

% emission_means:   1 by 2 by (m+1) array of mean vectors for the Gaussian 
%                   emission densities for each latent state

% emission_covs:    2 by 2 by (m+1) array of covariance matrices for
%                   the Gaussian emission densities for each latent state

% Author:           Sam Parsons
% Date created:     27/09/2016
% Last amended:     29/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

    isScalar = @(x) isnumeric(x) && isreal(x) && isscalar(x);
    isPositiveScalar = @(x) isScalar(x) && (x > 0);
    isMatrix = @(x) isnumeric(x) && isreal(x) && ismatrix(x);
    is3DMatrix = @(x) isnumeric(x) && isreal(x) && (ndims(x) == 3);
    
    if nargin < 4
        error('all 4 arguments must be input')
    end

    if ~(iscell(y) && iscolumn(y))
        tmpy{1} = y;
        y = tmpy;
        
        t_p_x{1} = p_x;
        p_x = t_p_x;
        
        t_p_x_x_next{1} = p_x_x_next;
        p_x_x_next = t_p_x_x_next;
    end
    

    n_people = length(y);
    dim_x = size(y{1}, 3);
    
    for person_idx = 1:n_people
        s_y = size(y{person_idx});

        % Check on y
        if ~(is3DMatrix(y{person_idx}) && all(s_y(2:3) == [2, dim_x]))
            error('each y must be a [n_p 2 dim_x] real array')
        end
        % Check on p_x
        if ~(isMatrix(p_x{person_idx}) && all(size(p_x{person_idx}) == [s_y(1), dim_x]))
            error('each p_x must be a [size(y, 1), size(y, 3)] real array')
        end
        
        num_tol = 1e-8;
        is_prob_dist = abs(1 - sum(p_x{person_idx}, 2)) < num_tol;
        if ~all(is_prob_dist)
            error('all rows of each p_x must be probability distributions')
        end
        
        % Check on p_x_x_next
        if ~(is3DMatrix(p_x_x_next{person_idx}) &&...
                all(size(p_x_x_next{person_idx}) == [dim_x, dim_x, s_y(1)-1]))
            error('each p_x_x_next must be a [dim_x, dim_x, size(y, 1)-1] real array')
        end
        is_prob_dist = arrayfun(@(row_idx)...
            all(all(p_x_x_next{person_idx}(:, :, row_idx) >= 0)) &&...
            (abs(1 - sum(sum(p_x_x_next{person_idx}(:, :, row_idx)))) < num_tol),...
            1:(s_y(1)-1));
        if ~all(is_prob_dist)
            error('all matrices of each p_x_x_next must be probability distributions')
        end
    end

    if ~(isPositiveScalar(min_eig))
        error('min_eig must be a positive real number')
    end
    
%     *********************************************************************

%     *********************************************************************
%     Main body of function. pi_1 is optimised by setting it to be
%     proportional to the sum over people of p_x{i}(1, :), summing to 1, P
%     is optimised by setting each row of it to be proportional to the sum
%     over people of sum(p_x_x_next{i}, 3), with each row summing to 1,
%     each mean vector j in emission_means is optimised by setting it to
%     equal the weighted sum over people and times of the corresponding
%     transformation j in y{i}, with each weight proportional to 
%     p_x{i}(t, j), and the weights for each j sum to 1. Each covariance
%     matrix j in emission_covs is optimised by setting it to be the
%     weighted sum of (y{i}(t, :, j) - mu(j))' * (y{i}(t, :, j) - mu(j)),
%     with weights as with emission_means optimisation. Each covariance
%     matrix computed for emission_covs hsa its eigenvalues thresholded by
%     min_eig
%     *********************************************************************

    fprintf('\tOptimising parameters...\n')
 
    
    pi_1 = sum(cell2mat(arrayfun(@(person_idx) p_x{person_idx}(1, :),...
        (1:n_people)', 'UniformOutput', false)), 1);
    pi_1 = pi_1 ./ sum(pi_1);
    
    P = sum(cell2mat(arrayfun(@(person_idx) sum(p_x_x_next{person_idx}, 3),...
        reshape(1:n_people, 1, 1, n_people), 'UniformOutput', false)), 3);
    P = bsxfun(@rdivide, P, sum(P, 2));
    
    weights_z = sum(cell2mat(arrayfun(@(person_idx) sum(p_x{person_idx}, 1),...
        (1:n_people)', 'UniformOutput', false)), 1);
    weights = cellfun(@(p_x_person)...
        reshape(bsxfun(@rdivide, p_x_person, weights_z), size(p_x_person, 1), 1, dim_x),...
        p_x, 'UniformOutput', false);
    
    emission_means = sum(cell2mat(arrayfun(@(person_idx)...
        sum(bsxfun(@times, y{person_idx}, weights{person_idx}), 1),...
        (1:n_people)', 'UniformOutput', false)), 1);
    emission_covs = zeros(2, 2, dim_x, n_people);
    
    for person_idx = 1:n_people
        n_p = size(y{person_idx}, 1);
        diffs = num2cell(bsxfun(@minus, y{person_idx}, emission_means), 2);
        diffs_sq = cellfun(@(diff_vec) diff_vec' * diff_vec, diffs,...
            'UniformOutput', false);
        diffs_sq = cell2mat(reshape(diffs_sq, 1, 1, n_p, dim_x));
        weights_person = reshape(weights{person_idx}, 1, 1, n_p, dim_x);
        emission_covs(:, :, :, person_idx) = squeeze(sum(bsxfun(@times, diffs_sq, weights_person), 3));
    end
    
    emission_covs = sum(emission_covs, 4);
    for x_idx = 1:size(emission_covs, 3)
        [V, D] = eig(emission_covs(:, :, x_idx));
        D = diag(max(diag(D), min_eig));
        emission_covs(:, :, x_idx) = V * D * V';
    end
    fprintf('\tOptimising parameters completed.\n')

end