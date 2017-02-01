% *************************************************************************
% Script for initialising the parameters needed to train an hmm.
% *************************************************************************

axis_dir = [[0; 1], [1; 1]./sqrt(2), [1; 0], [1; -1]./sqrt(2), [0; -1], [-1; -1]./sqrt(2), [-1; 0], [-1; 1]./sqrt(2)];
axis_orth_dir = [axis_dir(:, 3:end), axis_dir(:, 1:2)];
epsilon = 0.01;
diff_length = 4;
% integral(@(theta) 2\log(2) - log(1-cos(theta)), 0, pi) = 6.5328/2 (4 d.p.)
movement_z = 6.5328/2;



