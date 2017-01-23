function [ patient ] = interpolatePatientGaps( patient, msPeriod, intFx, intFy)
% Estimates the gaps in the eye tracking with a linear interpolation in
% both coordinates. Also adds field in struct listing which time stamps are
% blinks

for trial = 1:length(patient.trials),
    fprintf('Interpolate trial %d...\n', trial);
    tstamps = patient.trials{trial}.eye_positions(:, 1);
    xpos  = patient.trials{trial}.eye_positions(:, 2);
    ypos  = patient.trials{trial}.eye_positions(:, 3);
    
    % Calculate the timestapms missing
    tsmiss = getMissingTs(tstamps, msPeriod);
    
    fprintf('Interpolate x axis...\n');
    [~, xc] = intFx(tstamps, tsmiss, xpos);
    fprintf('Interpolate y axis...\n');
    [tsc, yc] = intFy(tstamps, tsmiss, ypos);
    
    patient.trials{trial}.eye_positions = [tsc xc yc];
    patient.trials{trial}.missing_ts = tsmiss;
end


