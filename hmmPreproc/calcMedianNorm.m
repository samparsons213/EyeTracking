function [ median_norm ] = calcMedianNorm( u )
%CALCMEDIANNORM Calculates the median norm by row of a matrix

median_norm = median(sqrt(sum(u.^2, 2)));

end

