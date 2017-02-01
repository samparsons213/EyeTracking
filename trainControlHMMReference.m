% This script trains the reference HMM for one control patient
% It considers that the data for the patient has been loaded and parsed
% please see (readAndParsePatients.m and gapInterpolation.m)

clear all
close all
addpath('./hmm/')
addpath('./hmmPreproc/')

% Load the information of interpolated patients (filename could change)
load 'interpolatedPatients.mat'

% Load the parameters to do preprocessing of a raw eye tracking data
initialiseHMMmodule

% Parameters for the training of HMM
n_iters = 500;
n_restarts = 20;
opt_epsilon = 1e-5;
min_eig = 1e-3;

% Get the experiments of the control patient

controlPatientName = 'lewwi';

% A = A(strcmp(A(:,4), 'T'), :);

for i = 1:length(patients),
    if(strcmp(patients{i}.name, controlPatientName)),
        selectedPatient = patients{i};
        break;
    end
end

% Number of different experiments to consider
n_exp = 12;
% We get the information for different trials, for only one reference control
y_tr = cellfun(@(trial) trial.eye_positions, selectedPatient.trials, 'UniformOutput', false);
target_tr = cellfun(@(trial) trial.targets, selectedPatient.trials, 'UniformOutput', false);


% We fit a model for each trial with multiple restarts
tic;
for idx = 1%:n_exp
    fprintf('Beginning parameter estimation for experiment: %d\n', idx)
    % Organise the input
    ts = y_tr{idx}(:,1);
    eye_tracking = y_tr{idx}(:,2:end);
    tst = target_tr{idx}(:, 1);
    target_placement = target_tr{idx}(:, 2:end);
    % Preprocess input
    [y, zero_prob, u] = hmmPreproc(ts, eye_tracking, tst, target_placement, axis_dir, axis_orth_dir, diff_length);
    
    % Do this once?
    median_norm_u = calcMedianNorm(u);
    
    % Calculate the conditional probabilities of movement directions given
    % each latent direction or 'no movement'
    p_ugx = condProbUX(u', axis_dir, median_norm_u, movement_z, epsilon);
    
    % Train HMM
    [pi_1(idx, :), P(:, :, idx), emission_means(idx, :, :), emission_covs(:, :, :, idx)] =...
        EM(y, p_ugx, zero_prob, n_iters, opt_epsilon, min_eig, n_restarts);
end

toc

%     [pi_1(idx, :), P(:, :, idx), emission_means(idx, :, :), emission_covs(:, :, :, idx)] =...
%         EM_ManyPeopleMultipleRestarts(y_tr{idx}, p_ugx_tr{idx}, double(zp_tr{idx}), 500, 1e-5, 20, 1e-3);
% dim_x = size(axis_dir, 2) + 1;
% [pi_1, P, emission_means, emission_covs] = generateRandomParameters(dim_x);

