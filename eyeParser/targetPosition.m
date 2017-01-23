function targetPos = targetPosition(line)
% Extracts the position of the target in an eye tracking experiment
% Example: 'MSG	1121020 -15  !V TARGET_POS TARG1 (960, 583) 1 0'
    elements = strsplit(line{1}, '[\s\(\),]+', 'DelimiterType', 'RegularExpression');
    targetPos(1) = str2double(elements(2));
    targetPos(2) = str2double(elements(7));
    targetPos(3) = str2double(elements(8));
end

