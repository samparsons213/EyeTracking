% % tic;
% % parfor idx = 1:600
% %     fprintf('Starting test %d\n', idx)
% %     SVMModel = fitcsvm(x_l2o_train(:, :, idx), y_l2o_train, 'ClassNames',[false,true]);
% %     preds_l2o(:, idx) = predict(SVMModel, x_l2o_test(:, :, idx));
% % % %     [~, w] = optimalAlpha(x_l2o(:, :, idx), y_l2o(:, :, idx), 1e-5, 2e5, 5e1);
% % % %     test_idx = row_map(idx, 2:3);
% % % %     dp = x_l2o_test(:, :, idx) * w;
% % % %     probs_l2o(:, :, idx) = bsxfun(@rdivide, exp(dp), sum(exp(dp), 2));
% % % %     preds_l2o(:, :, idx) = bsxfun(@eq, dp, max(dp, [], 2));
% %     fprintf('\tCompleted test %d\n', idx)
% % end
% % toc

% % tic;
% % parfor idx = 1:50
% %     fprintf('Starting test %d\n', idx)
% %     SVMModel = fitcsvm(x_l1o_train(:, :, idx), y_l1o_train(:, idx), 'ClassNames',[false,true]);
% %     preds_l1o(idx) = predict(SVMModel, s_fisher_fv_norm(:, idx)');
% % % %     [~, w] = optimalAlpha(x_l1o_train(:, :, idx), y_l1o_train(:, :, idx), 1e-5, 2e5, 5e1);
% % % %     dp = x(idx, :) * w;
% % % %     probs_l1o(idx, :) = exp(dp) ./ sum(exp(dp));
% % % %     preds_l1o(idx, :) = dp == max(dp);
% %     fprintf('\tCompleted test %d\n', idx)
% % end
% % toc

tic;
for idx = 1:500
    fprintf('Starting test %d\n', idx)
% %     train_idx(:, idx) = [sort(randperm(20, 10))'; 20+sort(randperm(30, 15))'];
% %     test_idx(:, idx) = setdiff((1:50)', train_idx(:, idx));
    x_train = s_fisher_fv_norm(:, train_idx(:, idx))';
    y_train = ~y(train_idx(:, idx), 1);
    x_test = s_fisher_fv_norm(:, test_idx(:, idx))';
    SVMModel = fitcsvm(x_train, y_train, 'ClassNames',[false,true]);
    preds_lho(:, idx) = predict(SVMModel, x_test);
% %     x_train = x(train_idx(:, idx), :);
% %     y_train = [y(train_idx(:, idx), 1), ~y(train_idx(:, idx), 1)];
% %     [~, w] = optimalAlpha(x_train, y_train, 1e-5, 2e5, 5e1);
% %     dp = x(test_idx(:, idx), :) * w;
% %     probs_lho(:, idx) = exp(dp(:, 2)) ./ sum(exp(dp), 2);
% %     preds_lho(:, idx) = dp(:, 2) == max(dp, [], 2);
% %     n_completed = idx;
% %     save('leave_half_out_x_validation_results.mat', 'n_completed',...
% %         'probs_lho', 'preds_lho', '-append')
    fprintf('\tCompleted test %d\n', idx)
end
toc