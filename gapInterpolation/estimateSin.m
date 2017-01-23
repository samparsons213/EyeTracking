function [tsc, xc] = estimateSin(ts, x)
% Interporlates a periodic X movement
    m_est = mean(x);
%     period = 10; 
    noise = 1;
%     sinAmpl = 0.5;
    SEamp = 1;
    SEvar = 1;
    displac = 5;

    meanfunc = {@meanConst}; hyp.mean = [m_est];
%     covfunc  = {@covSum, {@covCos, @covSEiso}}; hyp.cov = log([period, sinAmpl, SEamp, SEvar]);
    covfunc = {@covSum, {@covConst, @covSEiso}}; hyp.cov = log([displac, SEamp, SEvar]);
    likfunc  = @likGauss; sn = noise; hyp.lik = log(sn);
    % Optimize parameters
    [hyp, nlml] = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, ts, x);
    fprintf('Optimization of parameters from %f to %f\n', nlml(1), nlml(end));
    % Estimate missing
    tsmiss = getMissingTs(ts, 4);
    [mta, ~] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, ts, x, tsmiss);
    % Recompose signal
    [tsc, xc] = recompose(ts, x, tsmiss, mta);
end