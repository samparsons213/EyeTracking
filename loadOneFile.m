function [eye_tracking_data, target_data] = loadOneFile(file_folder,...
    file_name)
% Loads the eye tracking and target for one person's file.

% Inputs:

% file_folder:      character string giving the folder the data file is in

% file_name:        character string giving the name of the data file

% Outputs:

% eye_tracking_data:n_experiments by 2 cell array, each row of which
%                   contains the data for one experiment, and the first
%                   column contains the arrays of time stamps for each
%                   experiment, and the second column contains the xy
%                   coordinates of each eye track at each time stamp

% target_data:      n_experiments by 2 cell array, structured as per
%                   eye_tracking_data, but containing the data for the
%                   target in each experiment

% Author:           Sam Parsons
% Date created:     27/09/2016
% Last amended:     27/09/2016

%     *********************************************************************
%     Check inputs
%     *********************************************************************

    if nargin < 2
        error('Both arguments must be input')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of code. First, the number of experiments is counted by
%     running through the data file. Next, the number of eye tracks and the
%     number of target placements are counted by running through the file
%     again. Finally, the file is run through one last time, this time
%     recording the eye track events and the target events.
%     
%     Note that as blinking is not perfectly accounted for by the SBLINK
%     (L/R) and EBLINK (L/R) lines, a more pragmatic approach is taken to
%     identifying them. A blink is defined as any line(s) where
%     any(isnan(str2double(new_line(2:3)))) == 1. During each blink, the
%     eye tracking data for the times immediately previous and subsequent
%     are linearly interpolated between to obtain estimates of the eye
%     positions at each blink time stamp. Rows which have any(isnan(...
%     that come before the first correctly recorded row, or after the last
%     one, are ignored.
%     *********************************************************************

    fileID = fopen([file_folder, file_name]);
    if fileID == -1
        error('File failed to open')
    end
    fprintf('Successfully opened %s\n', file_name)
%     Count experiments
    fprintf('Counting number of experiments in file...\n')
    frewind(fileID)
    n_experiments = 0;
    while ~feof(fileID)
        new_line = fgetl(fileID);
        new_line = strsplit(new_line);
        if strcmp(new_line{1}, 'START')
            n_experiments = n_experiments + 1;
        end
    end
    fprintf('\tThere are %d experiments in the file\n', n_experiments)
%     Initialise variables
    eye_tracking_data = cell(n_experiments, 2);
    target_data = cell(n_experiments, 2);
    n_eye_tracks = zeros(n_experiments, 1);
    n_targets = zeros(n_experiments, 1);
    n_blinks = zeros(n_experiments, 1);
    any_nan = false(2, 1);
%     Count data events for each experiment
    fprintf('Counting number of data events in each experiment...\n')
    frewind(fileID)
    experiment_idx = 0;
    while ~feof(fileID)
        new_line = fgetl(fileID);
        new_line = strsplit(new_line);
        if strcmp(new_line{1}, 'START')
            experiment_idx = experiment_idx + 1;
            tracking_started = false;
        end
        end_line = new_line{end};
        is_data_line = ~isempty(end_line) &&...
            all(arrayfun(@(idx) strcmp(end_line(idx), '.'),...
            1:length(end_line)));
% %         (strcmp(new_line{end}, '.') || strcmp(new_line{end}, '...'))
        if is_data_line
            any_nan(1) = any(isnan(str2double(new_line(2:3))));
            if ~any_nan(1) && (n_eye_tracks(experiment_idx) == 0)
                tracking_started = true;
            end
            if tracking_started
                n_eye_tracks(experiment_idx) = n_eye_tracks(experiment_idx) + 1;
                if any_nan(1) && ~any_nan(2)
                    n_blinks(experiment_idx) = n_blinks(experiment_idx) + 1;
                end
            end
        end
        any_nan(2) = any_nan(1);
        if (length(new_line) == 10) && strcmp(new_line{5}, 'TARGET_POS')
            n_targets(experiment_idx) = n_targets(experiment_idx) + 1;
        end
    end
    for experiment_idx = 1:n_experiments
        fprintf('\tThere were %d eye tracks, %d blinks, and %d target placements in experiment %d\n',...
            n_eye_tracks(experiment_idx), n_blinks(experiment_idx),...
            n_targets(experiment_idx), experiment_idx)
        eye_tracking_data{experiment_idx, 1} =...
            zeros(n_eye_tracks(experiment_idx), 1);
        eye_tracking_data{experiment_idx, 2} =...
            zeros(n_eye_tracks(experiment_idx), 2);
        target_data{experiment_idx, 1} =...
            zeros(n_targets(experiment_idx), 1);
        target_data{experiment_idx, 2} =...
            zeros(n_targets(experiment_idx), 2);
    end
%     Record data
    fprintf('Extracting data events for each experiment from file...\n')
    frewind(fileID)
    experiment_idx = 0;
    eye_tracking_row_idx = 0;
    target_row_idx = 0;
    blink_details = cell(n_experiments, 1);
    while ~feof(fileID)
        new_line = fgetl(fileID);
        new_line = strsplit(new_line);
        if strcmp(new_line{1}, 'START')
            experiment_idx = experiment_idx + 1;
            tracking_started = false;
            eye_tracking_row_idx = 0;
            target_row_idx = 0;
            blink_details{experiment_idx} =...
                [zeros(n_blinks(experiment_idx), 1),...
                ones(n_blinks(experiment_idx), 1)];
            blink_idx = 0;
            any_nan = false(2, 1);
        end
        end_line = new_line{end};
        is_data_line = ~isempty(end_line) &&...
            all(arrayfun(@(idx) strcmp(end_line(idx), '.'),...
            1:length(end_line)));
% %         (strcmp(new_line{end}, '.') || strcmp(new_line{end}, '...'))
        if is_data_line
            any_nan(1) = any(isnan(str2double(new_line(2:3))));
            if ~any_nan(1) && (eye_tracking_row_idx == 0)
                tracking_started = true;
            end
            if tracking_started
                eye_tracking_row_idx = eye_tracking_row_idx + 1;
                if any_nan(1)
                    if ~any_nan(2)
                        blink_idx = blink_idx + 1;
                        blink_details{experiment_idx}(blink_idx, 1) =...
                            eye_tracking_row_idx;
                    else
                        blink_details{experiment_idx}(blink_idx, 2) =...
                            blink_details{experiment_idx}(blink_idx, 2) + 1;
                    end
                end
                eye_tracking_data{experiment_idx, 1}(eye_tracking_row_idx) =...
                    str2double(new_line{1});
                eye_tracking_data{experiment_idx, 2}(eye_tracking_row_idx, :) =...
                    str2double(new_line(2:3));
            end
        end
        any_nan(2) = any_nan(1);
        if (length(new_line) == 10) && strcmp(new_line{5}, 'TARGET_POS')
            target_row_idx = target_row_idx + 1;
            target_data{experiment_idx, 1}(target_row_idx) =...
                str2double(new_line{2});
            target_data{experiment_idx, 2}(target_row_idx, 1) =...
                str2double(new_line{7}(2:(end-1)));
            target_data{experiment_idx, 2}(target_row_idx, 2) =...
                str2double(new_line{8}(1:(end-1)));
        end
    end
%     retrospective blink interpolation
    for experiment_idx = 1:n_experiments
        for blink_idx = 1:n_blinks(experiment_idx)
            blink_start = blink_details{experiment_idx}(blink_idx, 1);
            blink_stop = blink_details{experiment_idx}(blink_idx, 1) +...
                blink_details{experiment_idx}(blink_idx, 2) - 1;
            pre_blink =...
                eye_tracking_data{experiment_idx, 2}((blink_start-1), :);
            if blink_stop < length(eye_tracking_data{experiment_idx, 1})
                post_blink =...
                    eye_tracking_data{experiment_idx, 2}((blink_stop+1), :);
                blink_length = blink_details{experiment_idx}(blink_idx, 2);
                for blink_idx2 = 0:(blink_length-1)
                    eye_tracking_data{experiment_idx, 2}((blink_start+blink_idx2), :) =...
                        pre_blink +...
                        ((blink_idx2+1)/(blink_length+1)) .* (post_blink-pre_blink);
                end
            else
                eye_tracking_data{experiment_idx, 1} =...
                    eye_tracking_data{experiment_idx, 1}(1:(blink_start-1));
                eye_tracking_data{experiment_idx, 2} =...
                    eye_tracking_data{experiment_idx, 2}(1:(blink_start-1), :);
            end
        end
    end
    fprintf('\tData extraction completed\n')

end