function targetPos = eyePosition(line)
% Extracts the position of the gaze in an eye tracking experiment
% Example: '1115636	  888.6	  -72.5	  287.0	.'
    elements = strsplit(line{1});
    targetPos(1) = str2double(elements(1));
    targetPos(2) = str2double(elements(2));
    targetPos(3) = str2double(elements(3));
end