% Script for extracting eye tracking and target data from file

% Preliminaries
file_folder = 'C:\Users\Sam\OneDrive\C-PLACID\Eye Tracking Data\';
file_name = 'armgr.asc';
first_data_line = 138;

% Open file, read line by line. If line is an eye track then append to eye
% tracking data, if line is a target placement then place in target data,
% otherwise skip line. Line is determined to be an eye track if its length
% after splitting on ' ' is 5 (timestamp, x, y, dilation, .), and the
% finial element is a '.'. Line is determined to be a target placement if
% length after splitting on ' ' is 10, and the fifth element is 'TARET_POS'

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
fprintf('Counting number of data events for each experiment in file...\n')
while ~feof(fileID)
    new_line = fgetl(fileID);
    new_line = strsplit(new_line);
    if strcmp(new_line{1}, 'END')
        experiment_idx = experiment_idx + 1;
    end
    if (length(new_line) == 5) && strcmp(new_line{end}, '.')
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
fprintf('Extracting data events for each experiment from file...\n')
while ~feof(fileID)
    new_line = fgetl(fileID);
    new_line = strsplit(new_line);
    if strcmp(new_line{1}, 'END')
        experiment_idx = experiment_idx + 1;
        eye_track_row_idx = 0;
        target_placing_row_idx = 0;
    end
    if (length(new_line) == 5) && strcmp(new_line{end}, '.')
        eye_track_row_idx = eye_track_row_idx + 1;
        eye_tracking{experiment_idx, 1}(eye_track_row_idx) =...
            str2double(new_line{1});
        eye_tracking{experiment_idx, 2}(eye_track_row_idx, 1) =...
            str2double(new_line{2});
        eye_tracking{experiment_idx, 2}(eye_track_row_idx, 2) =...
            str2double(new_line{3});
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