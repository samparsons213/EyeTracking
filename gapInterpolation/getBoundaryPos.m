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