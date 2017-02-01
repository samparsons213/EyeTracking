function p_ugx = condProbUX(u, axis_dir, median_norm_u, movement_z, epsilon)
% Returns conditional probabilities of the target direction u
% given each of the latent directions and the 'no movement' latent state

% Inputs:

% u:            2 by n target direction

% axis_dir:     2 by m array of latent directions, where each column is a
%               unit vector

% median_norm_u: positive real number giving the median norm of the control
%                directions in the experiment

% movement_z:   positive real giving the normalising constant for the
%               conditional distributions p(u | x) for x = 2,...,n_l_dirs.
%               Should equal
%               integral(@(theta) 2\log(2) - log(1-cos(theta)), 0, 2*pi) =
%               6.5328 (4 d.p.)

% epsilon:      real number in (0, 0.5) giving the closest acceptable
%               norm-difference (see below) to 0 and 1

% Outputs:

% p_ugx:        n by (1+m) vector of conditional densities, with the first
%               element giving the conditional density for the 'no movement'
%               latent state, and the last m elements giving the
%               conditional densities of the corresponding columns of
%               axis_dir given the control direction u. Note that this is not
%               a probability distribution, just a vector of conditional
%               densities, each element from a different conditional
%               distribution

% Author:       Sam Parsons, David Martinez
% Date created: 20/09/2016
% Last amended: 23/09/2016

%     *********************************************************************
%     Check input arguments
%     *********************************************************************

    % Numerical tolerance
    num_tol = 1e-8;
    % Utils
    isNumMatrix = @(x) isnumeric(x) && ismatrix(x);
    isPosScalar = @(x) isnumeric(x) && isreal(x) && isscalar(x) && (x > 0);
    
    if nargin < 5
        error('all 5 arguments must be input')
    end

    if ~(isnumeric(u) && (size(u, 1) == 2)),
        error('u must be numeric column vector of size [2 1]')
    end
   
    if ~(isNumMatrix(axis_dir) && (size(axis_dir, 1) == 2))
        error('axis_dir must be a 2 by n numeric array for some positive integer n')
    end
    
    norms = sqrt(sum(axis_dir.^2));
    if ~all(abs(norms-1) < num_tol),
        error('all columns in axis_dir must be unit vectors')
    end
    
    if ~isPosScalar(median_norm_u)
        error('median_norm_u must be a positive real scalar')
    end
    
    if ~isPosScalar(movement_z)
        error('movement_z must be a positive real scalar')
    end
   
    if ~(isPosScalar(epsilon) && (epsilon < 0.5))
        error('epsilon must be a real scalar in (0, 0.5)')
    end
    
    %     *********************************************************************

    %     *********************************************************************
    %     Main body of code. Produces a conditional density of the control
    %     direction u for each of the latent directions, and for the 'no
    %     movement' latent state. These densities will then be used to weight
    %     the transition matrix before it is renormalised (row by row). 
    %     For the latent directions, we take the norm of the difference between
    %     (normalised) u and each column of axis_dir, and half them so they're
    %     all <= 1 and their logs are all non-positive. These are then
    %     thresholded to be at least epsilon and at most 1-epsilon, to avoid
    %     log(0) and log(1). Thresholded values are logged and negated, and
    %     then normalised by dividing by movement_z. For the 'no movement'
    %     latent state, the norm of u as a fraction of median_norm_u is
    %     negatively logged (after thresholding as before), and normalised by
    %     dividing by median_norm_u
    %     *********************************************************************

    n_l_dir = size(axis_dir, 2);
    N = size(u, 2);
    
    % Calculate norm of all target directions
    normU = sqrt(sum(u.^2));
    % Normalize target directions
    u = bsxfun(@rdivide, u, normU);
    
    % Allocate output variable
    exp_scores = zeros(N, 1+n_l_dir);
    
    % Calculate the score for 'no movement'
    exp_scores(:, 1) = min(1, normU' / median_norm_u);
    
    % Do the difference on each axis (remember that first is reserved for
    % 'no movement' state) and calculate the score
    for k=1:n_l_dir,
        exp_scores(:, 1+k) = 0.5*sqrt(sum(bsxfun(@minus, u, axis_dir(:, k)).^2))';
    end
    
    % We need this to have it in the open interval (0, 1)
    exp_scores = min(max(epsilon, exp_scores), 1-epsilon);
    
    Z = [median_norm_u, repmat(movement_z, 1, n_l_dir)];
    p_ugx = bsxfun(@rdivide, -log(exp_scores), Z);