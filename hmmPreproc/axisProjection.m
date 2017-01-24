function [projectedMovements, zero_prob] = axisProjection(gazePos, axis_dir, axis_orth_dir, diff_length)
% Transforms the T by 2 eye tracking data for each experiment for one
% person into the form needed for inference in the proposed HMM. Difference
% vectors of a specified size are computed from eye_tracking data, and
% projected onto the orthogonal basis implied by each canonical direction
% in axis_dir. When the projection onto the primary basis vector (i.e. the
% canonical direction itself) for each basis is non-positive, the
% corresponding element of the logical array zero_prob is set to false to
% indicate that its corresponding emission density is 0.

% Inputs:

% gazePos:      a T by 2 array of eye tracking locations

% axis_dir:     2 by n array of canonical latent directions, where each
%               column is a unit vector

% axis_orth_dir: 2 by n array of vectors orthogonal to each of the canonical
%                latent directions, where each column is a unit vector

% diff_length:  positive integer giving the gaps between eye tracks for
%               computing difference vectors, e.g. if diff_length = 3 and
%               T = 7 for some element i of gazePos, then the difference vectors
%               (prior to any transformations) would be
%               gazePos{i}(4, :) - gazePos{i}(1, :); gazePos{i}(7, :)-gazePos{i}(4, :). If not
%               input then its default value is 1

% Outputs:

% gazePos:      a (x-1) by 2 by (n+1) array of transformed difference vectors, where
%               x=length(1:diff_length:T)=1+floor((T-1)/diff_length)

% zero_prob:    n_experiments by 1 cell, where each element is a (x-1) by n
%               binary array, where zeros correspond to zero posterior
%               probability for that direction at that time, and ones
%               correspond to positive posterior probability for that
%               direction at that time, and x is as in gazePos above

% Author:       Sam Parsons
% Date created: 20/09/2016
% Last amended: 12/10/2016

%  *********************************************************************
%     Check input arguments
%  *********************************************************************
    % Get the number of experiments
    n_experiments = length(gazePos);
    % Numerical tolerance
    num_tol = 1e-8;
    % Util functions
    isNumMatrix = @(x) isnumeric(x) && ismatrix(x);
    isIntScalar = @(x) isnumeric(x) && isreal(x) && isscalar(x) && (x - round(x) < num_tol);
    
    if nargin < 3
        error('the first 3 arguments must be input')
    end

    if ~(isNumMatrix(gazePos) && ...
          size(gazePos, 2) == 2 && ...
          size(gazePos, 1) > 1) 
        error('gazePos must be a n by 2 matrix for some positive integer n')
    end
    
    if ~(isNumMatrix(axis_dir) && (size(axis_dir, 1) == 2))
        error('axis_dir must be a 2 by n numeric array for some positive integer n')
    end
    
    if ~(isNumMatrix(axis_orth_dir) && all(size(axis_orth_dir) == size(axis_dir)))
        error('axis_orth_dir must be a numeric array of the same size as axis_dir')
    end
    
    norms = sqrt(sum(axis_dir.^2));
    if ~all(abs(norms-1) < num_tol),
        error('all columns in axis_dir must be unit vectors')
    end
    
    norms = sqrt(sum(axis_orth_dir.^2));
    if ~all(abs(norms-1) < num_tol),
        error('all columns in axis_orth_dir must be unit vectors')
    end
    
    if ~all(diag(abs(axis_dir'*axis_orth_dir)) < num_tol)
        error('all columns of axis_orth_dir must be orthogonal to the corresponding column of axis_dir')
    end
    
    % diff_length must be a positive integer no greater than the smallest length of a trial. 
    % If outside the acceptable range, it is placed inside at the nearest boundary. 
    % If not provided, the default value is 1
    if nargin == 3
        diff_length = 1;
    else
        if ~isIntScalar(diff_length),
            error('diff_length must be a scalar integer')
        end
        smallest_trial = size(gazePos, 1);
        diff_length = min(max(1, diff_length), smallest_trial-1);
    end
    
    % *********************************************************************

    % *********************************************************************
    % Main body of code. Perform differentiation on gazePos, then the transforming
    % required for each latent direction. Record which latent directions
    % will have zero posterior probability (<=> non-positive movement in that direction).
    % 
    % The transformation consists on measuring how much a movement is
    % aligned with one axis (we use the log since we assume that its distribution is lognormal)
    % and how much it is tilted in the orthogonal direction. So each movement
    % generates 2 components on each axis introduced.
    % There is an extra "no-movement" component with the original movement
    % vector.
    % *********************************************************************

    n_l_dirs  = size(axis_dir, 2); 

    % Differentiate the trace
    y_diff = movementDiff(gazePos, diff_length);

    % Preallocate out tensor for this experiment
    y_transformed = zeros(size(y_diff, 1), 2, 1+n_l_dirs);

    %No-movement component
    y_transformed(:,:,1) = y_diff;

    % Log on the axis movement
    x1 = y_diff * axis_dir;
    zero_prob = x1 > 0;
    x1(zero_prob) = log(x1(zero_prob));
    % Tilt on the othogonal directions
    x2 = y_diff * axis_orth_dir;

    % Add axis movement components
    y_transformed(:, 1, 2:end) = x1;
    y_transformed(:, 2, 2:end) = x2;

    % Add pattern
    projectedMovements = y_transformed;

end