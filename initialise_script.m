% *************************************************************************
% Script for loading data for one person, and transforming the data into
% form needed for proposed HMM. Also writes l_dirs and orth_l_dirs
% variables
% *************************************************************************

l_dirs = [[0; 1], [1; 1]./sqrt(2), [1; 0], [1; -1]./sqrt(2), [0; -1], [-1; -1]./sqrt(2), [-1; 0], [-1; 1]./sqrt(2)];
orth_l_dirs = [l_dirs(:, 3:end), l_dirs(:, 1:2)];
process_file_script
[y, zero_prob] = yTransformHMM(eye_tracking(:, 2), l_dirs, orth_l_dirs);
control_seq_hmm = controlSequenceHMM(eye_tracking, target_placing);
epsilon = 0.01;
p_ugx = condProbUXAll(control_seq_hmm, l_dirs, epsilon);
dim_x = size(l_dirs, 2) + 1;
[pi_1, P, emission_means, emission_covs] = generateRandomParameters(dim_x);