function [eye_tracking] = preprocessPatient(file_name)
% Script for extracting eye tracking and target data from file

% Read all the content of the file in memory
fileID = fopen(file_name);
file_content = textscan(fileID, '%s', 'delimiter', '\n');
lines = file_content{1};
fclose(fileID);

% Regular expressions of different lines
START_REGEXP = 'MSG\s\d+\sTRIALID\s\d+';
SCREEN_REGEXP = 'MSG\s+\d+\s+GAZE_COORDS\s+.+';
TARGET_REGEXP = 'MSG\s+\d+\s+.+\s+.+\s+\TARGET_POS';
EYE_REGEXP = '\d+\s+[-+]?[0-9]*\.?[0-9]+.+';
BLINK_REGEXP = 'EBLINK\s.+';
SACC_REGEXP = 'ESACC\s.+';
NAME_REGEXP = '\w*(?=\.asc)';

% Count the number of experiments
fprintf('Counting number of experiments in file...\n')
startTrials = findLines(lines, START_REGEXP);
n_trials = length(startTrials);
fprintf('There are %d experiments in the file\n', n_trials)

[sn, en] = regexp(file_name, NAME_REGEXP);
eye_tracking.name = file_name(sn:en);
eye_tracking.trials = cell(n_trials, 1);

fprintf('Parsing %s\n', eye_tracking.name)

% Parsing each experiment
for i = 1:n_trials,
    fprintf('Parsing trial %d\n', i)
    startTrial = startTrials(i);
    if (i == length(startTrials))
        endTrial = length(lines);
    else
        endTrial = startTrials(i+1)-1;
    end
    trial_lines = lines(startTrial:endTrial);
    
    % Parse dimensions of the screen
    gaze_line = trial_lines(findLines(trial_lines, SCREEN_REGEXP));
    gaze_elements = strsplit(gaze_line{1});
    eye_tracking.trials{i}.dimensions = arrayfun(@(x) str2double(x), gaze_elements(4:7));
    
    % Parse positions of the target
    target_lines = trial_lines(findLines(trial_lines, TARGET_REGEXP));
    eye_tracking.trials{i}.targets = cell2mat(arrayfun(@extractTargetPos, target_lines, 'UniformOutput',  false));
    
    % Parse eye_tracking positions and pupil dilatations
    eye_lines = trial_lines(findLines(trial_lines, EYE_REGEXP));
    eye_tracking.trials{i}.eye_positions = cell2mat(arrayfun(@extractEyePosition, eye_lines, 'UniformOutput',  false));
    eye_tracking.trials{i}.eye_pupil = cell2mat(arrayfun(@extractPupilDilatation, eye_lines, 'UniformOutput',  false));
    % Parse blinks
    blink_lines = trial_lines(findLines(trial_lines, BLINK_REGEXP));
    eye_tracking.trials{i}.blinks = cell2mat(arrayfun(@extractBlink, blink_lines, 'UniformOutput',  false));
    % Parse saccades detected by the apparatus
    sacc_lines = trial_lines(findLines(trial_lines, SACC_REGEXP));
    eye_tracking.trials{i}.saccades = cell2mat(arrayfun(@extractSaccade, sacc_lines, 'UniformOutput',  false));
end

end

function lines = findLines(lines_content, regExp)
    % Check if regex is present
    chechPresent = @(x) any(x==1);
    lines = find(cellfun(chechPresent, regexp(lines_content, regExp)));
end

function targetPos = extractTargetPos(line)
    % Example: 'MSG	1121020 -15  !V TARGET_POS TARG1 (960, 583) 1 0'
    elements = strsplit(line{1}, '[\s\(\),]+', 'DelimiterType', 'RegularExpression');
    targetPos(1) = str2double(elements(2));
    targetPos(2) = str2double(elements(7));
    targetPos(3) = str2double(elements(8));
end

function targetPos = extractEyePosition(line)
    % Example: '1115636	  888.6	  -72.5	  287.0	.'
    elements = strsplit(line{1});
    targetPos(1) = str2double(elements(1));
    targetPos(2) = str2double(elements(2));
    targetPos(3) = str2double(elements(3));
end

function targetPos = extractPupilDilatation(line)
    % Example: '1115636	  888.6	  -72.5	  287.0	.'
    elements = strsplit(line{1});
    targetPos(1) = str2double(elements(1));
    targetPos(2) = str2double(elements(4));
end

function targetPos = extractBlink(line)
    % Example: 'EBLINK R 946944	947024	84'
    elements = strsplit(line{1});
    targetPos(1) = str2double(elements(3));
    targetPos(2) = str2double(elements(4));
end

function targetPos = extractSaccade(line)
    % Example: ESACC R  937284	937324	44	  954.1	  527.8	  554.9	  513.8	  10.53	    481
    elements = strsplit(line{1});
    targetPos(1) = str2double(elements(3));
    targetPos(2) = str2double(elements(4));
    targetPos(2) = str2double(elements(6));
    targetPos(2) = str2double(elements(7));
    targetPos(2) = str2double(elements(8));
    targetPos(2) = str2double(elements(9));
end

