function [tr_sb, tr_fk] =...
    testFittedClassificationModel3(gram_matrix, distances, train_rate,...
    n_ctrl, n_ad, n_pca, ad_strat)
    
%     ad_start determines how ad cases are dealt with in training - 0
%     places them with pca, 1 places them with ctrl, 2 ignores them
%     altogether

%     Preliminaries
    n_patients = n_ctrl + n_ad + n_pca;
    train_rate = max(train_rate, 0);
    train_rate = min(train_rate, 1);
    
    n_train_ctrl = min(n_ctrl-1, ceil(train_rate * n_ctrl));
    n_train_ad = min(n_ad-1, ceil(train_rate * n_ad));
    n_train_pca = min(n_pca-1, ceil(train_rate * n_pca));
    
    n_test_ctrl = n_ctrl - n_train_ctrl;
    n_test_ad = n_ad - n_train_ad;
    n_test_pca = n_pca - n_train_pca;
    
%     Build random training and test sets
    train_idx_ctrl = randperm(n_ctrl, n_train_ctrl);
    train_idx_ad = n_ctrl + randperm(n_ad, n_train_ad);
    train_idx_pca = n_ctrl + n_ad + randperm(n_pca, n_train_pca);
    if ad_strat < 2
        train_idx = [train_idx_ctrl, train_idx_ad, train_idx_pca];
    else
        train_idx = [train_idx_ctrl, train_idx_pca];
    end
    n_train = length(train_idx);
    n_test = n_patients - n_train;
    n_fi = size(gram_matrix, 3);
    x_train = cat(2, ones(n_train, 1, n_fi), gram_matrix(train_idx, train_idx, :));
    
    y_ctrl = [true(n_ctrl, 1); false(n_ad + n_pca, 1)];
    y_ad = [false(n_ctrl, 1); true(n_ad, 1); false(n_pca, 1)];
    y_pca = [false(n_ctrl + n_ad, 1); true(n_pca, 1)];
    
%     Fit sparse Bayes classifiers
    if ad_strat == 0
        y = y_ad | y_pca;
    else
        y = y_pca;
    end
    y_train = y(train_idx);

    sbparms = struct('Relevant', {}, 'Value', {});
    for exp_idx = 1:n_fi
        sbparms(exp_idx) = SparseBayes('Bernoulli',...
            x_train(:, :, exp_idx), y_train);
    end
    
%     Make sparse Bayes class predictions on test set
    test_idx = setdiff(1:n_patients, train_idx);
    x_test = cat(2, ones(n_test, 1, n_fi), gram_matrix(test_idx, train_idx, :));
    
    test_probs = cell2mat(arrayfun(@(exp_idx)...
        1 ./ (1 + exp(-x_test(:, sbparms(exp_idx).Relevant, exp_idx) * sbparms(exp_idx).Value)),...
        1:n_fi, 'UniformOutput', false));
    test_probs = mean(test_probs > 0.5, 2);
    y_preds_sb = test_probs > 0.5;
    
%     Breakdown of sparse Bayes test results
    y_test_ctrl = y_ctrl(test_idx, :, 1);
    y_test_ad = y_ad(test_idx, :, 1);
    y_test_pca = y_pca(test_idx, :, 1);
    
    tr_sb = [sum(y_test_ctrl & ~y_preds_sb), sum(y_test_ctrl & y_preds_sb);...
        sum(y_test_ad & ~y_preds_sb), sum(y_test_ad & y_preds_sb);...
        sum(y_test_pca & ~y_preds_sb), sum(y_test_pca & y_preds_sb)];

%     Make fisher kernel distance predictions on test set
    train_idx_not_ctrl = setdiff(train_idx, train_idx_ctrl);
    test_dists = cat(2, mean(distances(test_idx, train_idx_ctrl, :), 2),...
        mean(distances(test_idx, train_idx_not_ctrl, :), 2));
    [~, y_preds_fk] = min(test_dists, [], 2);
    y_preds_fk = mean(y_preds_fk - 1, 3) == 1;

%     Breakdown of fisher kernel distance test results
    tr_fk = [sum(y_ctrl(test_idx) & ~y_preds_fk), sum(y_ctrl(test_idx) & y_preds_fk);...
        sum(y_ad(test_idx) & ~y_preds_fk), sum(y_ad(test_idx) & y_preds_fk);...
        sum(y_pca(test_idx) & ~y_preds_fk), sum(y_pca(test_idx) & y_preds_fk)];

end