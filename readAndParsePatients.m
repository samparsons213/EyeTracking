% Loads all the patients data from the files and parses all the trials
% The format is the SmartEye hardware format

% The eye tracking data has missing segments due to blinks and errors in
% the capture of data. The interpolation to be applied is configurable
% between:

clear all;
close all;

% Add parsing primitives
addpath('./eyeParser');

%% Configuration section
% This is the folder where the patient data resides
folder = '../SmoothPersuitResults/';
outFile = 'parsedPatients.mat';

%% Get all the files in the folder specified and filter those that are 
%  experiment recordings
files = dir(folder);
count = 1;
while(count < length(files)),
    file = files(count);
    if(isempty(regexp(file.name, '\.asc', 'ONCE')))
        files(count) = [];
    else
        count = count+1;
    end
end

%% Load the experiments, parse and interpolate
fprintf('%d patients to be processed:\n', count);

patients = cell(1, count);
for f=1:count,
    file = files(f);
    patients{f} = parsePatient(strcat(folder, file.name));
    % Save current progress
    save(outFile, 'patients');
end

fprintf('%d patients correctly processed:\n', count);
for i=1:length(patients),
    fprintf('%s\n', patients{i}.name);
end
