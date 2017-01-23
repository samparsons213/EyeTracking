function [tsc, xc] = estimateLinear(ts, tsmiss, x)
% Linearly interpolates the X movement

    % Preliminaries
    [gap_starts, gap_stops] = getMissingTsStartsStops(tsmiss, 4);
    [boundary_starts, boundary_stops] = getBoundaryPos(gap_starts,...
        gap_stops, ts, x);
    mta = zeros(size(tsmiss));
    stop = 0;
    % Interpolation
    for gap_idx = 1:length(gap_starts),
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