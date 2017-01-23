function targetPos = saccadeEvent(line)
% Extracts a saccade event from an eye tracking experiment
% Example: ESACC R  937284	937324	44	  954.1	  527.8	  554.9	  513.8	  10.53	    481
    elements = strsplit(line{1});
    targetPos(1) = str2double(elements(3));
    targetPos(2) = str2double(elements(4));
    targetPos(2) = str2double(elements(6));
    targetPos(2) = str2double(elements(7));
    targetPos(2) = str2double(elements(8));
    targetPos(2) = str2double(elements(9));
end