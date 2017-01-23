function [tscomp] = getMissingTs(ts, N)
% Extracts the timestamps that are missing with the assumption that the
% machine takes a sample every N milliseconds
    tscomp = min(ts):N:max(ts);
    tscomp(ismember(tscomp,ts)) = [];
    tscomp = tscomp';
end