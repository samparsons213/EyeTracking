% Loads all the patients and parses all the trials
% clear all;
close all;

load 'parsedPatients.mat';

for patient=1:length(patients),
    fprintf('Smoothing %s...\n', patients{patient}.name);
    if(~isempty(patients{patient}.trials))
%         patients{patient} = estimatePatientGapsSin(patients{patient});
        patients{patient} = estimatePatientGapsLinInterp(patients{patient});
    end
    % Save to file
    fprintf('Saving data\n');
%     save('smoothedPatients.mat', 'patients');
%     save('smoothingWS.mat', 'patient');
end
    save('interpolatedPatients.mat', 'patients');
    save('interpolatingWS.mat', 'patient');
