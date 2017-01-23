function [eye_tracking] = parsePatient(file_name)
% Script for extracting eye tracking and target data from file

% Read all the content of the file in memory
fileID = fopen(file_name);
file_content = textscan(fileID, '%s', 'delimiter', '\n');
lines = file_content{1};
fclose(fileID);

% Regular expressions of different lines that are part of the format
START_REGEXP = 'MSG\s\d+\sTRIALID\s\d+';
SCREEN_REGEXP = 'MSG\s+\d+\s+GAZE_COORDS\s+.+';
TARGET_REGEXP = 'MSG\s+\d+\s+.+\s+.+\s+\TARGET_POS';
EYE_REGEXP = '\d+\s+[-+]?[0-9]*\.?[0-9]+.+';
BLINK_REGEXP = 'EBLINK\s.+';
SACC_REGEXP = 'ESACC\s.+';
NAME_REGEXP = '\w*(?=\.asc)';

% Count the number of trials in the experiment
fprintf('Counting number of experiments in file...\n')
startTrials = filterLines(lines, START_REGEXP);
n_trials = length(startTrials);
fprintf('There are %d experiments in the file\n', n_trials)

[sn, en] = regexp(file_name, NAME_REGEXP);
eye_tracking.name = file_name(sn:en);
eye_tracking.trials = cell(n_trials, 1);

fprintf('Parsing %s\n', eye_tracking.name)

% Parsing each trial in the experiment
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
    gaze_line = trial_lines(filterLines(trial_lines, SCREEN_REGEXP));
    gaze_elements = strsplit(gaze_line{1});
    eye_tracking.trials{i}.dimensions = arrayfun(@(x) str2double(x), gaze_elements(4:7));
    
    % Parse positions of the target
    target_lines = trial_lines(filterLines(trial_lines, TARGET_REGEXP));
    eye_tracking.trials{i}.targets = cell2mat(arrayfun(@targetPosition, target_lines, 'UniformOutput',  false));
    
    % Parse eye_tracking positions and pupil dilatations
    eye_lines = trial_lines(filterLines(trial_lines, EYE_REGEXP));
    eye_tracking.trials{i}.eye_positions = cell2mat(arrayfun(@eyePosition, eye_lines, 'UniformOutput',  false));
    eye_tracking.trials{i}.eye_pupil = cell2mat(arrayfun(@pupilDilation, eye_lines, 'UniformOutput',  false));
    % Parse blinks
    blink_lines = trial_lines(filterLines(trial_lines, BLINK_REGEXP));
    eye_tracking.trials{i}.blinks = cell2mat(arrayfun(@blinkEvent, blink_lines, 'UniformOutput',  false));
    % Parse saccades detected by the apparatus
    sacc_lines = trial_lines(filterLines(trial_lines, SACC_REGEXP));
    eye_tracking.trials{i}.saccades = cell2mat(arrayfun(@saccadeEvent, sacc_lines, 'UniformOutput',  false));
end



