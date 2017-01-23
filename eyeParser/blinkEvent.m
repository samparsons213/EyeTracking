function targetPos = blinkEvent(line)
% Extracts a blink event from an eye tracking experiment
% Example: 'EBLINK R 946944	947024	84'
    elements = strsplit(line{1});
    targetPos(1) = str2double(elements(3));
    targetPos(2) = str2double(elements(4));
end