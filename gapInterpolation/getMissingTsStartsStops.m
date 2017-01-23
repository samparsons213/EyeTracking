function [gap_starts, gap_stops] = getMissingTsStartsStops(tsmiss, N)
% Decomposes sequence of missing timestamps into contiguous parts.
% see also getMissingTs(ts, N)

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