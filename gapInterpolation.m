% Interpolates the gaps that appear in the patients eye tracking due to
% blinks and errors in measurement

clear all;
close all;
addpath('./gapInterpolation/');

% Load the file with the raw recordings (see readAndParsePatients.m)
load 'parsedPatients.mat';

%% Configuration

% Configure the type of interpolation:
%    @estimateLinear : linear
%    @estimateSin: gaussian process interpolation of a sinusoidal component
%    @estimateConst: gaussian process interpolation of constant component

interFx = @estimateLinear;
interFy = @estimateLinear;
% Period of recording in ms
msPeriod = 4;
% File to save the patients interpolated
outFile = 'interpolatedPatients.mat';


%% Interpolation of gaps
for patient=1:length(patients),
    fprintf('Interpolating %s...\n', patients{patient}.name);
    if(~isempty(patients{patient}.trials)),
        patients{patient} = interpolatePatientGaps(patients{patient}, msPeriod, interFx, interFy);      
    end
    % Save to file
    save(outFile, 'patients');
end
fprintf('Interpolation finished\n');
