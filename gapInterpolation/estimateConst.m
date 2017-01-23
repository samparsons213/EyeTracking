function [tsc, yc] = estimateConst(ts, y)
% Interpolates a constant Y movement
    m_est = mean(y);
    displac = 5;
    SEamp = 1;
    SEvar = 1;
    noise = 1;

    meanfunc = {@meanConst}; hyp.mean = [m_est]; 
    covfunc = {@covSum, {@covConst, @covSEiso}}; hyp.cov = log([displac, SEamp, SEvar]);
    likfunc = @likGauss; sn = noise; hyp.lik = log(sn);

    % Optimize parameters
    [hyp, nlml] = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, ts, y);
    fprintf('Optimization of parameters from %f to %f\n', nlml(1), nlml(end));
    % Estimate missing
    tsmiss = getMissingTs(ts, 4);
    [mta, ~] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, ts, y, tsmiss);
    % Recompose signal
    [tsc, yc] = recompose(ts, y, tsmiss, mta);
end