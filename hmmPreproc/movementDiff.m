function [ d ] = movementDiff( gazePos, diff_length )
%MOVEMENTDIFF Makes the differentiation of a gaze trace

    d = diff(gazePos(1:diff_length:end, :));
end

