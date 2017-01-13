function [ patient ] = estimatePatientGapsLinInterp( patient )
% Estimates the gaps in the eye tracking with a linear interpolation in
% both coordinates. Also adds field in struct listing which time stamps are
% blinks

for trial = 1:length(patient.trials),
    fprintf('Smoothing trial %d...\n', trial);
    tstamps = patient.trials{trial}.eye_positions(:, 1);
    xpos  = patient.trials{trial}.eye_positions(:, 2);
    ypos  = patient.trials{trial}.eye_positions(:, 3);
    
    fprintf('Smoothing x axis %d...\n', trial);
    [~, xc] = estimateX(tstamps, xpos);
    fprintf('Smoothing y axis %d...\n', trial);
    [tsc, yc] = estimateY(tstamps, ypos);
    tsmiss = getMissingTs(tstamps, 4);
    
%     if(1),
%         figure
%         plot(patient.trials{trial}.eye_positions(:, 2), patient.trials{trial}.eye_positions(:, 3), '--*r')
%     end
    
    patient.trials{trial}.eye_positions = [tsc xc yc];
    patient.trials{trial}.missing_ts = tsmiss;
    
%     if(1),
%         hold on
%         plot(patient.trials{trial}.eye_positions(:, 2), patient.trials{trial}.eye_positions(:, 3), '-*b')
%     end
end

end

function [tsc, xc] = estimateX(ts, x)
% Linearly interpolates the X movement
    % Preliminaries
    tsmiss = getMissingTs(ts, 4);
    [gap_starts, gap_stops] = getMissingTsStartsStops(tsmiss, 4);
    [boundary_starts, boundary_stops] = getBoundaryPos(gap_starts,...
        gap_stops, ts, x);
    mta = zeros(size(tsmiss));
    stop = 0;
    % Interpolation
    for gap_idx = 1:length(gap_starts)
        start = stop + 1;
        stop = start + (gap_stops(gap_idx) - gap_starts(gap_idx)) / 4;
        start_stop = start:stop;
        mta(start_stop) = boundary_starts(gap_idx) +...
            ((start_stop-(start-1)) ./ (2 + stop - start)) .*...
            (boundary_stops(gap_idx) - boundary_starts(gap_idx));
    end
    % Recompose signal
    [tsc, xc] = recompose(ts, x, tsmiss, mta);
end


function [tsc, yc] = estimateY(ts, y)
% Linearly interpolates the X movement
    % Preliminaries
    tsmiss = getMissingTs(ts, 4);
    [gap_starts, gap_stops] = getMissingTsStartsStops(tsmiss, 4);
    [boundary_starts, boundary_stops] = getBoundaryPos(gap_starts,...
        gap_stops, ts, y);
    mta = zeros(size(tsmiss));
    stop = 0;
    % Interpolation
    for gap_idx = 1:length(gap_starts)
        start = stop + 1;
        stop = start + (gap_stops(gap_idx) - gap_starts(gap_idx)) / 4;
        start_stop = start:stop;
        mta(start_stop) = boundary_starts(gap_idx) +...
            ((start_stop-(start-1)) ./ (2 + stop - start)) .*...
            (boundary_stops(gap_idx) - boundary_starts(gap_idx));
    end
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

function [gap_starts, gap_stops] = getMissingTsStartsStops(tsmiss, N)
% Decomposes sequence of missing timestamps into contiguous parts with the
% assumption that the sequence of missing timestamps was obtained from
% getMissingTs(ts, N)
    if isempty(tsmiss)
        gap_starts = [];
        gap_stops = [];
    else
        stop_idx = [find(diff(tsmiss) > N); length(tsmiss)];
        start_idx = [1; stop_idx(1:(end-1))+1];
        try
            gap_starts = tsmiss(start_idx);
        catch ME
            rethrow ME
        end
        gap_stops = tsmiss(stop_idx);
    end
end

function [boundary_starts, boundary_stops] = getBoundaryPos(gap_starts,...
    gap_stops, ts, pos)
% Returns the positions of eye in given coordinate at the boundaries of
% each contiguous block of missing timestamps. boundary_starts gives the
% positions at the timestamp before the starts of the gaps, boundary_stops
% refer gives the positions at the timestamp after the ends of the gaps
    boundary_starts_idx =...
        arrayfun(@(idx) find(ts < gap_starts(idx), 1, 'last'),...
        1:length(gap_starts));
    boundary_stops_idx =...
        arrayfun(@(idx) find(ts > gap_stops(idx), 1, 'first'),...
        1:length(gap_stops));
    boundary_starts = pos(boundary_starts_idx);
    boundary_stops = pos(boundary_stops_idx);
end


