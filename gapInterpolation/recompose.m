function [x, y] = recompose(xor, yor, xint, yint)
% Recomposes a full signal from the real observations xor, yor
% and the estimated ones that interpolate xint, yint
    x = [xor; xint];
    y = [yor;yint];
    [x,I] = sort(x);
    y = y(I);
end