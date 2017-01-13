tr = zeros(3, 3, 500);
tic;
for idx = 1:500
    fprintf('Test run %d\n', idx)
    train_idx_ctrl = randperm(18, 9);
    train_idx_ad = randperm(24, 12);
    train_idx_pca = randperm(6, 3);
    train_idx1 = [train_idx_ctrl, train_idx_ad+18, train_idx_pca+42];
    train_idx2 = testing_group(train_idx1);
    x_train = [ones(24, 1), x_fip_sum(train_idx2, train_idx2)];
    y_train = y(train_idx2, :);
    [~, w] = optimalAlpha(x_train, y_train, 1e-4, 5e4, 5e2);
    test_idx = testing_group(setdiff(1:48, train_idx1));
    x_test = [ones(24, 1), x_fip_sum(test_idx, train_idx2)];
    y_test = y(test_idx, :);
    dp = x_test * w;
    preds = bsxfun(@eq, dp, max(dp, [], 2));
    tr(:, :, idx) = double(y_test)' * double(preds);
end
toc