function [test_results_sb, test_results_sb2, test_results_svm] =...
    testFittedClassificationModel(fisher_fv_ctrl, fisher_fv_ad,...
    fisher_fv_pca, train_rate)

    train_rate = max(train_rate, 0);
    train_rate = min(train_rate, 1);
    
    s_ctrl = size(fisher_fv_ctrl);
    n_train_ctrl = min(s_ctrl(2)-1, ceil(s_ctrl(2) * train_rate));
    train_idx_ctrl = randperm(s_ctrl(2), n_train_ctrl);
    if ndims(fisher_fv_ctrl) == 3
        x_train_ctrl = reshape(fisher_fv_ctrl(:, train_idx_ctrl, :),...
            s_ctrl(1), n_train_ctrl * s_ctrl(3))';
    else
        x_train_ctrl = fisher_fv_ctrl(:, train_idx_ctrl)';
    end
    n_train_ctrl = size(x_train_ctrl, 1);
    
    s_ad = size(fisher_fv_ad);
    n_train_ad = min(s_ad(2)-1, ceil(s_ad(2) * train_rate));
    train_idx_ad = randperm(s_ad(2), n_train_ad);
    if ndims(fisher_fv_ad) == 3
        x_train_ad = reshape(fisher_fv_ad(:, train_idx_ad, :),...
            s_ad(1), n_train_ad * s_ad(3))';
    else
        x_train_ad = fisher_fv_ad(:, train_idx_ad)';
    end    
    n_train_ad = size(x_train_ad, 1);
    
    s_pca = size(fisher_fv_pca);
    n_train_pca = min(s_pca(2)-1, ceil(s_pca(2) * train_rate));
    train_idx_pca = randperm(s_pca(2), n_train_pca);
    if ndims(fisher_fv_pca) == 3
        x_train_pca = reshape(fisher_fv_pca(:, train_idx_pca, :),...
            s_pca(1), n_train_pca * s_pca(3))';
    else
        x_train_pca = fisher_fv_pca(:, train_idx_pca)';
    end
    n_train_pca = size(x_train_pca, 1);
    
    x_train_pre = [x_train_ctrl; x_train_ad; x_train_pca];
    n_train = size(x_train_pre, 1);
    
    x_train = [ones(n_train, 1), x_train_pre];
    y_train = [false(n_train_ctrl, 1); true(n_train_ad + n_train_pca, 1)];
    sbparms = SparseBayes('Bernoulli', x_train, y_train);

    x_train2 = [ones(n_train, 1), x_train_pre * x_train_pre'];
    sbparms2 = SparseBayes('Bernoulli', x_train2, y_train);
    
    x_train3 = x_train(:, sbparms.Relevant);
    SVMModel = fitcsvm(x_train3, y_train, 'Standardize', true,...
        'ClassNames',[false,true]);

    test_idx_ctrl = setdiff(1:s_ctrl(2), train_idx_ctrl);
    if ndims(fisher_fv_ctrl) == 3
        x_test_ctrl = reshape(fisher_fv_ctrl(:, test_idx_ctrl, :),...
            s_ctrl(1), length(test_idx_ctrl) * s_ctrl(3))';
    else
        x_test_ctrl = fisher_fv_ctrl(:, test_idx_ctrl)';
    end
    n_test_ctrl = size(x_test_ctrl, 1);
    
    test_idx_ad = setdiff(1:s_ad(2), train_idx_ad);
    if ndims(fisher_fv_ad) == 3
        x_test_ad = reshape(fisher_fv_ad(:, test_idx_ad, :),...
            s_ad(1), length(test_idx_ad) * s_ad(3))';
    else
        x_test_ad = fisher_fv_ad(:, test_idx_ad)';
    end
    n_test_ad = size(x_test_ad, 1);
    
    test_idx_pca = setdiff(1:s_pca(2), train_idx_pca);
    if ndims(fisher_fv_pca) == 3
        x_test_pca = reshape(fisher_fv_pca(:, test_idx_pca, :),...
            s_pca(1), length(test_idx_pca) * s_pca(3))';
    else
        x_test_pca = fisher_fv_pca(:, test_idx_pca)';
    end
    n_test_pca = size(x_test_pca, 1);
    
    x_test_pre = [x_test_ctrl; x_test_ad; x_test_pca];
    n_test = size(x_test_pre, 1);
    x_test = [ones(n_test, 1), x_test_pre];
    x_test = x_test(:, sbparms.Relevant);
    x_test2 = [ones(n_test, 1), x_test_pre * x_train_pre'];
    x_test2 = x_test2(:, sbparms2.Relevant);
    
    y_ad = [false(n_test_ctrl, 1); true(n_test_ad, 1);...
        false(n_test_pca, 1)];
    y_pca = [false(n_test_ctrl + n_test_ad, 1); true(n_test_pca, 1)];
    y_test = y_ad | y_pca;
    
    sb_preds = 1 ./ (1 + exp(-x_test * sbparms.Value)) > 0.5;
    test_results_sb = [sum(~y_test & ~sb_preds), sum(~y_test & sb_preds);...
        sum(y_ad & ~sb_preds), sum(y_ad & sb_preds);...
        sum(y_pca & ~sb_preds), sum(y_pca & sb_preds)];

    sb_preds2 = 1 ./ (1 + exp(-x_test2 * sbparms2.Value)) > 0.5;
    test_results_sb2 = [sum(~y_test & ~sb_preds2), sum(~y_test & sb_preds2);...
        sum(y_ad & ~sb_preds2), sum(y_ad & sb_preds2);...
        sum(y_pca & ~sb_preds2), sum(y_pca & sb_preds2)];

%     lrparms = glmfit(x_train, y_train, 'binomial');
%     lr_preds = 1 ./ (1 + exp(-[ones(size(x_test, 1), 1), x_test] * lrparms)) > 0.5;
%     test_results.lrc = [sum(~y_test & ~lr_preds), sum(~y_test & lr_preds);...
%         sum(y_test & ~lr_preds), sum(y_test & lr_preds)];
% 
%     lrparms = glmfit(x_train, y_train, 'binomial', 'constant', 'off');
%     lr_preds = 1 ./ (1 + exp(-x_test * lrparms)) > 0.5;
%     test_results.lr = [sum(~y_test & ~lr_preds), sum(~y_test & lr_preds);...
%         sum(y_test & ~lr_preds), sum(y_test & lr_preds)];
    
    svm_preds = predict(SVMModel, x_test);    
    test_results_svm = [sum(~y_test & ~svm_preds), sum(~y_test & svm_preds);...
        sum(y_ad & ~svm_preds), sum(y_ad & svm_preds);...
        sum(y_pca & ~svm_preds), sum(y_pca & svm_preds)];

end