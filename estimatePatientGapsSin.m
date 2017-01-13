function [ patient ] = estimatePatientGapsSin( patient )
% Estimates the gaps in the eye tracking with a GP
% The covariance function is tailored to a periodic experiment
% in the x dimension and static in the y dimension

for trial = 1:length(patient.trials),
    fprintf('Smoothing trial %d...\n', trial);
    tstamps = patient.trials{trial}.eye_positions(:, 1);
    xpos  = patient.trials{trial}.eye_positions(:, 2);
    ypos  = patient.trials{trial}.eye_positions(:, 3);
    
    fprintf('Smoothing x axis %d...\n', trial);
    [~, xc] = estimateX(tstamps, xpos);
    fprintf('Smoothing y axis %d...\n', trial);
    [tsc, yc] = estimateY(tstamps, ypos);
    
%     if(1),
%         figure
%         plot(patient.trials{trial}.eye_positions(:, 2), patient.trials{trial}.eye_positions(:, 3), '--*r')
%     end
    
    patient.trials{trial}.eye_positions = [tsc xc yc];
    
%     if(1),
%         hold on
%         plot(patient.trials{trial}.eye_positions(:, 2), patient.trials{trial}.eye_positions(:, 3), '-*b')
%     end
end

end

function [tsc, xc] = estimateX(ts, x)
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


function [tsc, yc] = estimateY(ts, y)
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

function [tscomp] = getMissingTs(ts, N)
% Extracts the timestamps that are missing with the assumption that the
% machine takes a sample every N milliseconds
    tscomp = min(ts):N:max(ts);
    tscomp(ismember(tscomp,ts)) = [];
    tscomp = tscomp';
end

function [x, y] = recompose(xor, yor, xint, yint)
% Recomposes a full signal from the real observations xor, yor
% and the estimated ones that interpolate xint, yint
    x = [xor; xint];
    y = [yor;yint];
    [x,I] = sort(x);
    y = y(I);
end


