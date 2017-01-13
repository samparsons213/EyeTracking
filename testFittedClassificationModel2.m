function [tr_sb, tr_fk] =...
    testFittedClassificationModel2(gram_matrix, distances, train_rate,...
    n_ctrl, n_ad, n_pca)

%     Preliminaries
    n_patients = n_ctrl + n_ad + n_pca;
    train_rate = max(train_rate, 0);
    train_rate = min(train_rate, 1);
    
    n_train_ctrl = min(n_ctrl-1, ceil(train_rate * n_ctrl));
    n_train_ad = min(n_ad-1, ceil(train_rate * n_ad));
    n_train_pca = min(n_pca-1, ceil(train_rate * n_pca));
    n_train = n_train_ctrl + n_train_ad + n_train_pca;
    
    n_test_ctrl = n_ctrl - n_train_ctrl;
    n_test_ad = n_ad - n_train_ad;
    n_test_pca = n_pca - n_train_pca;
    n_test = n_patients - n_train;
    
%     Build random training and test sets
    train_idx_ctrl = randperm(n_ctrl, n_train_ctrl);
    train_idx_ad = n_ctrl + randperm(n_ad, n_train_ad);
    train_idx_pca = n_ctrl + n_ad + randperm(n_pca, n_train_pca);
    train_idx = [train_idx_ctrl, train_idx_ad, train_idx_pca];
    x_train = [ones(n_train, 1), gram_matrix(train_idx, train_idx)];
    
    y_ctrl = [true(n_ctrl, 1); false(n_ad + n_pca, 1)];
    y_ad = [false(n_ctrl, 1); true(n_ad, 1);...
        false(n_pca, 1)];
    y_pca = [false(n_ctrl + n_ad, 1); true(n_pca, 1)];
    
%     Fit sparse Bayes classifiers for each diagnosis class
    sbparms_ctrl = SparseBayes('Bernoulli', x_train, y_ctrl(train_idx));
    sbparms_ad = SparseBayes('Bernoulli', x_train, y_ad(train_idx));
    sbparms_pca = SparseBayes('Bernoulli', x_train, y_pca(train_idx));
    
%     Make sparse Bayes class predictions on test set
    test_idx = setdiff(1:n_patients, train_idx);
    x_test = [ones(n_test, 1), gram_matrix(test_idx, train_idx)];
    
    ctrl_probs = 1 ./ (1 + exp(-x_test(:, sbparms_ctrl.Relevant) * sbparms_ctrl.Value));
    ad_probs = 1 ./ (1 + exp(-x_test(:, sbparms_ad.Relevant) * sbparms_ad.Value));
    pca_probs = 1 ./ (1 + exp(-x_test(:, sbparms_pca.Relevant) * sbparms_pca.Value));
    [~, y_preds_sb] = max([ctrl_probs, ad_probs, pca_probs], [], 2);
    
%     Breakdown of sparse Bayes test results
    y_ctrl = y_ctrl(test_idx);
    y_ad = y_ad(test_idx);
    y_pca = y_pca(test_idx);
    
    tr_sb = [sum(y_ctrl & (y_preds_sb == 1)), sum(y_ctrl & (y_preds_sb == 2)),...
        sum(y_ctrl & (y_preds_sb == 3)); sum(y_ad & (y_preds_sb == 1)),...
        sum(y_ad & (y_preds_sb == 2)), sum(y_ad & (y_preds_sb == 3));...
        sum(y_pca & (y_preds_sb == 1)), sum(y_pca & (y_preds_sb == 2)),...
        sum(y_pca & (y_preds_sb == 3))];
    
%     Make fisher kernel distance predictions on test set
    test_dists = [mean(distances(test_idx, train_idx_ctrl), 2),...
        mean(distances(test_idx, train_idx_ad), 2),...
        mean(distances(test_idx, train_idx_pca), 2)];
    [~, y_preds_fk] = min(test_dists, [], 2);

%     Breakdown of fisher kernel distance test results
    tr_fk = [sum(y_ctrl & (y_preds_fk == 1)), sum(y_ctrl & (y_preds_fk == 2)),...
        sum(y_ctrl & (y_preds_fk == 3)); sum(y_ad & (y_preds_fk == 1)),...
        sum(y_ad & (y_preds_fk == 2)), sum(y_ad & (y_preds_fk == 3));...
        sum(y_pca & (y_preds_fk == 1)), sum(y_pca & (y_preds_fk == 2)),...
        sum(y_pca & (y_preds_fk == 3))];

end