n_trials = 10000;
tr_fk1 = zeros(3, 2, n_trials);
tr_fk2 = zeros(3, 2, n_trials);
fisher_info_inv = fisher_info3 \ eye(size(fisher_info, 1));
% % fisher_info_pinv = pinv(fisher_info3);
training_rate = 0.7;
print_rate = 500;

n_train_ctrl = ceil(n_ctrl * training_rate);
n_train_ad = ceil(n_ad * training_rate);
n_train_pca = ceil(n_pca * training_rate);
n_train = n_train_ctrl + n_train_ad + n_train_pca;
n_test_ctrl = n_ctrl - n_train_ctrl;
n_test_ad = n_ad - n_train_ad;
n_test_pca = n_pca - n_train_pca;
n_test = n_test_ctrl + n_test_ad + n_test_pca;
n_patients = n_ctrl + n_ad + n_pca;

n_trials = 3;
n_total = n_trials * n_patients;

fisher_fv_all = reshape(cell2mat(arrayfun(@(exp_idx) [fisher_fv(:, :, exp_idx),...
    fisher_fv_ad(:, :, exp_idx), fisher_fv_pca(:, :, exp_idx)],...
    reshape(1:n_trials, 1, 1, n_trials), 'UniformOutput', false)),...
    size(fisher_fv, 1), n_total)';
gram_matrix_all = fisher_fv_all * fisher_info_inv * fisher_fv_all';

distances = sqrt(cell2mat(arrayfun(@(idx1) arrayfun(@(idx2)...
    (fisher_fv_all(idx1, :) - fisher_fv_all(idx2, :)) * fisher_info_inv *...
    (fisher_fv_all(idx1, :) - fisher_fv_all(idx2, :))', 1:n_total),...
    (1:n_total)', 'UniformOutput', false)));

train_idx_ctrl = zeros(1, n_train_ctrl*n_trials);
train_idx_ad = zeros(1, n_train_ad*n_trials);
train_idx_pca = zeros(1, n_train_pca*n_trials);
test_idx_ctrl = zeros(1, n_test_ctrl*n_trials);
test_idx_ad = zeros(1, n_test_ad*n_trials);
test_idx_pca = zeros(1, n_test_pca*n_trials);

for idx = 1:n_trials
    if mod(idx, print_rate) == 0
        fprintf('Trial: %d\n', idx)
    end
    train_idx_ctrl(1:n_train_ctrl) = randperm(n_ctrl, n_train_ctrl);
    test_idx_ctrl(1:n_test_ctrl) =...
        setdiff(1:n_ctrl, train_idx_ctrl(1:n_train_ctrl));
    train_idx_ctrl((n_train_ctrl+1):end) = [train_idx_ctrl(1:n_train_ctrl)+n_patients,...
        train_idx_ctrl(1:n_train_ctrl)+(2*n_patients)];
    test_idx_ctrl((n_test_ctrl)+1:end) = [test_idx_ctrl(1:n_test_ctrl)+n_patients,...
        test_idx_ctrl(1:n_test_ctrl)+(2*n_patients)];

    train_idx_ad(1:n_train_ad) = randperm(n_ad, n_train_ad) + n_ctrl;
    test_idx_ad(1:n_test_ad) =...
        setdiff(1:n_ad, train_idx_ad(1:n_train_ad) - n_ctrl) + n_ctrl;
    train_idx_ad((n_train_ad+1):end) = [train_idx_ad(1:n_train_ad)+n_patients,...
        train_idx_ad(1:n_train_ad)+(2*n_patients)];
    test_idx_ad((n_test_ad)+1:end) = [test_idx_ad(1:n_test_ad)+n_patients,...
        test_idx_ad(1:n_test_ad)+(2*n_patients)];

    train_idx_pca(1:n_train_pca) = randperm(n_pca, n_train_pca) + n_ctrl + n_ad;
    test_idx_pca(1:n_test_pca) =...
        setdiff(1:n_pca, train_idx_pca(1:n_train_pca) - n_ctrl - n_ad) + n_ctrl + n_ad;
    train_idx_pca((n_train_pca+1):end) = [train_idx_pca(1:n_train_pca)+n_patients,...
        train_idx_pca(1:n_train_pca)+(2*n_patients)];
    test_idx_pca((n_test_pca)+1:end) = [test_idx_pca(1:n_test_pca)+n_patients,...
        test_idx_pca(1:n_test_pca)+(2*n_patients)];
    
    train_idx = [train_idx_ctrl, train_idx_ad, train_idx_pca];
    test_idx = [test_idx_ctrl, test_idx_ad, test_idx_pca];
    
    test_dists_ctrl =...
        mean(distances([test_idx_ctrl, test_idx_ad, test_idx_pca], train_idx_ctrl), 2);
    test_dists_ad =...
        mean(distances([test_idx_ctrl, test_idx_ad, test_idx_pca], train_idx_ad), 2);
    test_dists_pca =...
        mean(distances([test_idx_ctrl, test_idx_ad, test_idx_pca], train_idx_pca), 2);
    [~, y_pred_fk1] =...
        min([test_dists_ctrl, test_dists_ad, test_dists_pca], [], 2);
    y_pred_fk1 = logical(y_pred_fk1 - 1);

    x_train =...
        [ones(length(train_idx), 1), gram_matrix_all(train_idx, train_idx)];
    y_train_ad = [false(length(train_idx_ctrl), 1); true(length(train_idx_ad), 1);...
        false(length(train_idx_pca), 1)];
    y_train_pca = [false(length(train_idx_ctrl) + length(train_idx_ad), 1);...
        true(length(train_idx_pca), 1)];
    y_train = y_train_ad | y_train_pca;
    
    x_test = [ones(length(test_idx), 1), gram_matrix_all(test_idx, train_idx)];
    y_test_ad = [false(length(test_idx_ctrl), 1); true(length(test_idx_ad), 1);...
        false(length(test_idx_pca), 1)];
    y_test_pca = [false(length(test_idx_ctrl) + length(test_idx_ad), 1);...
        true(length(test_idx_pca), 1)];
    y_test = y_test_ad | y_test_pca;
    
    sbparms = SparseBayes('Bernoulli', x_train, y_train);
    y_pred_fk2 = SB2_Sigmoid(x_test(:, sbparms.Relevant) * sbparms.Value) > 0.5;
    
    tr_fk1(:, :, idx) = [sum(~y_test & ~y_pred_fk1), sum(~y_test & y_pred_fk1);...
        sum(y_test_ad & ~y_pred_fk1), sum(y_test_ad & y_pred_fk1);...
        sum(y_test_pca & ~y_pred_fk1), sum(y_test_pca & y_pred_fk1)];

    tr_fk2(:, :, idx) = [sum(~y_test & ~y_pred_fk2), sum(~y_test & y_pred_fk2);...
        sum(y_test_ad & ~y_pred_fk2), sum(y_test_ad & y_pred_fk2);...
        sum(y_test_pca & ~y_pred_fk2), sum(y_test_pca & y_pred_fk2)];
end