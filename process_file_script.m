% Script for extracting eye tracking and target data from file

% Preliminaries
file_folder = 'C:\Users\Sam\OneDrive\C-PLACID\Eye Tracking Data\';
% % file_name = 'armgr.asc'; % Control patient
% % first_data_line = 138; % Specific to armgr.asc
% % % % file_name = 'conpa.asc'; % PCA patient
% % % % first_data_line = 36; % Specific to conpa.asc
file_name = 'biged.asc'; % Control patient
first_data_line = 91; % Specific to biged.asc

n_eyes = 1; % Specific to some data files. Number of eye coordinates in
%             each track - either one (left or right) or two (left and 
%             right)

% % n_eyes = 2; % Specific to some data files

eye_lr = 'L'; % Specific to some data files

% % eye_lr = 'R'; % Specific to some data files

eye_track_line_length = 2 + 3*n_eyes; % timestamp, final full stop, and
%                                       x,y,dilation for each eye 
%                                       coordinate per track
eye_track_x = 2 + (eye_lr == 'R') * (n_eyes-1)*3; % if both eyes are
%                                                   tracked then left eye 
%                                                   is recorded first, so 
%                                                   coordinates on line 
%                                                   will be shifted 3 
%                                                   places to the right for
%                                                   right eye when 
%                                                   n_eyes = 2

% Open file, read line by line. If line is an eye track then append to eye
% tracking data, if line is a target placement then place in target data,
% otherwise skip line. Line is determined to be an eye track if its length
% after splitting on ' ' is eye_track_line_length (timestamp, x, y,
% dilation, . - see above for two eye equiivalent), and the final element
% is a '.'. Line is determined to be a target placement if length after
% splitting on ' ' is 10, and the fifth element is 'TARET_POS'. If a SBLINK
% L/R (start blink) line is read, don't read eye tracks until an EBLINK L/R
% (end blink) line is read. Only eye blinks for one eye are considered as
% only one eye is being analysed right now.

% % cd(file_folder)
fileID = fopen([file_folder, file_name]);
n_experiments = 0;
% Initial pass to count number of experiments, then another pass to count
% the number of eye tracks and target placements in each experiment. First
% pass through the non-data lines at the beginning of the file. After
% counting is complete, initialise arrays for storing data.
fprintf('Skipping first %d lines...\n', first_data_line-1)
for line_skip = 1:(first_data_line-1)
    if ~feof(fileID)
        new_line = fgetl(fileID);
    else
        error('end of file reached before first_data_line')
    end
end
fprintf('Counting number of experiments in file...\n')
while ~feof(fileID)
    new_line = fgetl(fileID);
    new_line = strsplit(new_line);
    if strcmp(new_line{1}, 'END')
        n_experiments = n_experiments + 1;
    end
end
fprintf('There are %d experiments in the file\n', n_experiments)
eye_tracking = cell(n_experiments, 2);
target_placing = cell(n_experiments, 2);
n_eye_tracks = zeros(n_experiments, 1);
n_targets = zeros(n_experiments, 1);

frewind(fileID)
fprintf('Skipping first %d lines...\n', first_data_line-1)
for line_skip = 1:(first_data_line-1)
    if ~feof(fileID)
        new_line = fgetl(fileID);
    else
        error('end of file reached before first_data_line')
    end
end
experiment_idx = 1;
add_eye_track = true;
fprintf('Counting number of data events for each experiment in file...\n')
while ~feof(fileID)
    new_line = fgetl(fileID);
    new_line = strsplit(new_line);
    if strcmp(new_line{1}, 'END')
        experiment_idx = experiment_idx + 1;
    end
    if strcmp(new_line{1}, 'SBLINK') && strcmp(new_line{2}, eye_lr)
        add_eye_track = false;
    end
    if strcmp(new_line{1}, 'EBLINK') && strcmp(new_line{2}, eye_lr)
        add_eye_track = true;
    end
    if (length(new_line) == eye_track_line_length) && strcmp(new_line{end}, '.') && add_eye_track
        n_eye_tracks(experiment_idx) = n_eye_tracks(experiment_idx) + 1;
    end
    if (length(new_line) == 10) && strcmp(new_line{5}, 'TARGET_POS')
        n_targets(experiment_idx) = n_targets(experiment_idx) + 1;
    end
end
for experiment_idx = 1:n_experiments
    fprintf('There were %d eye tracks and %d target placements in experiment %d\n',...
        n_eye_tracks(experiment_idx), n_targets(experiment_idx), experiment_idx)
    eye_tracking{experiment_idx, 1} = zeros(n_eye_tracks(experiment_idx), 1);
    eye_tracking{experiment_idx, 2} = zeros(n_eye_tracks(experiment_idx), 2);
    target_placing{experiment_idx, 1} = zeros(n_targets(experiment_idx), 1);
    target_placing{experiment_idx, 2} = zeros(n_targets(experiment_idx), 2);
end

% Pass through file again, this time storing relevant data in appropriate
% arrays
frewind(fileID)
fprintf('Skipping first %d lines...\n', first_data_line-1)
for line_skip = 1:(first_data_line-1)
    if ~feof(fileID)
        new_line = fgetl(fileID);
    else
        error('end of file reached before first_data_line')
    end
end
experiment_idx = 1;
eye_track_row_idx = 0;
target_placing_row_idx = 0;
add_eye_track = true;
fprintf('Extracting data events for each experiment from file...\n')
while ~feof(fileID)
    new_line = fgetl(fileID);
    new_line = strsplit(new_line);
    if strcmp(new_line{1}, 'END')
        experiment_idx = experiment_idx + 1;
        eye_track_row_idx = 0;
        target_placing_row_idx = 0;
    end
    if strcmp(new_line{1}, 'SBLINK') && strcmp(new_line{2}, eye_lr)
        add_eye_track = false;
    end
    if strcmp(new_line{1}, 'EBLINK') && strcmp(new_line{2}, eye_lr)
        add_eye_track = true;
    end
    if (length(new_line) == eye_track_line_length) && strcmp(new_line{end}, '.') && add_eye_track
        eye_track_row_idx = eye_track_row_idx + 1;
        eye_tracking{experiment_idx, 1}(eye_track_row_idx) =...
            str2double(new_line{1});
        eye_tracking{experiment_idx, 2}(eye_track_row_idx, 1) =...
            str2double(new_line{eye_track_x});
        eye_tracking{experiment_idx, 2}(eye_track_row_idx, 2) =...
            str2double(new_line{eye_track_x+1});
    end
    if (length(new_line) == 10) && strcmp(new_line{5}, 'TARGET_POS')
        target_placing_row_idx = target_placing_row_idx + 1;
        target_placing{experiment_idx, 1}(target_placing_row_idx) =...
            str2double(new_line{2});
        target_placing{experiment_idx, 2}(target_placing_row_idx, 1) =...
            str2double(new_line{7}(2:(end-1)));
        target_placing{experiment_idx, 2}(target_placing_row_idx, 2) =...
            str2double(new_line{8}(1:(end-1)));
    end
end