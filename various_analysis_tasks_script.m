vp_vs_bp = cell(n_people, 1);
for person_idx = 1:n_people
    vp = viterbiPath(y_exp1{person_idx}, p_ugx_exp1{person_idx}, pi_1_hat,...
        P_hat, emission_means_hat, emission_covs_hat,...
        zero_prob_exp1{person_idx});
    [~, bp] = max(p_ugx_exp1{person_idx}, [], 2);
    vp_vs_bp{person_idx} = cell2mat(arrayfun(@(vp_idx) arrayfun(@(bp_idx)...
        sum((vp == vp_idx) & (bp == bp_idx)),...
        1:size(p_ugx_exp1{person_idx}, 2)),...
        (1:size(p_ugx_exp1{person_idx}, 2))', 'UniformOutput', false));
end
vp_vs_bp_ad = cell(n_ad_patients, 1);
for person_idx = 1:n_ad_patients
    vp = viterbiPath(y_ad_exp1{person_idx}, p_ugx_ad_exp1{person_idx}, pi_1_hat,...
        P_hat, emission_means_hat, emission_covs_hat,...
        zero_prob_ad_exp1{person_idx});
    [~, bp] = max(p_ugx_ad_exp1{person_idx}, [], 2);
    vp_vs_bp_ad{person_idx} = cell2mat(arrayfun(@(vp_idx) arrayfun(@(bp_idx)...
        sum((vp == vp_idx) & (bp == bp_idx)),...
        1:size(p_ugx_ad_exp1{person_idx}, 2)),...
        (1:size(p_ugx_ad_exp1{person_idx}, 2))', 'UniformOutput', false));
end
vp_vs_bp_pca = cell(n_pca_patients, 1);
for person_idx = 1:n_pca_patients
    vp = viterbiPath(y_pca_exp1{person_idx}, p_ugx_pca_exp1{person_idx}, pi_1_hat,...
        P_hat, emission_means_hat, emission_covs_hat,...
        zero_prob_pca_exp1{person_idx});
    [~, bp] = max(p_ugx_pca_exp1{person_idx}, [], 2);
    vp_vs_bp_pca{person_idx} = cell2mat(arrayfun(@(vp_idx) arrayfun(@(bp_idx)...
        sum((vp == vp_idx) & (bp == bp_idx)),...
        1:size(p_ugx_pca_exp1{person_idx}, 2)),...
        (1:size(p_ugx_pca_exp1{person_idx}, 2))', 'UniformOutput', false));
end
exp_plot_idx = 1;
for person_idx = 1:n_people
    figure
    plot(eye_tracking_data{person_idx}{exp1_idx(person_idx) + (exp_plot_idx-1), 2}(:, 1),...
        eye_tracking_data{person_idx}{exp1_idx(person_idx) + (exp_plot_idx-1), 2}(:, 2), 'o')
% %     plot(y{person_idx}{exp1_idx(person_idx) + (exp_plot_idx-1)}(:, 1, 1),...
% %         y{person_idx}{exp1_idx(person_idx) + (exp_plot_idx-1)}(:, 2, 1), 'o')
    ax = gca();
    ax_min = min(ax.XLim(1), ax.YLim(1));
    ax_max = max(ax.XLim(2), ax.YLim(2));
    ax.XLim = [ax_min, ax_max];
    ax.YLim = [ax_min, ax_max];
end
for person_idx = 1:n_ad_patients
    figure
    plot(eye_tracking_data_ad{person_idx}{exp1_idx_ad(person_idx) + (exp_plot_idx-1), 2}(:, 1),...
        eye_tracking_data_ad{person_idx}{exp1_idx_ad(person_idx) + (exp_plot_idx-1), 2}(:, 2), 'o')
    ax = gca();
    ax_min = min(ax.XLim(1), ax.YLim(1));
    ax_max = max(ax.XLim(2), ax.YLim(2));
    ax.XLim = [ax_min, ax_max];
    ax.YLim = [ax_min, ax_max];
end
for person_idx = 1:n_pca_patients
    figure
    plot(eye_tracking_data_pca{person_idx}{exp1_idx_pca(person_idx) + (exp_plot_idx-1), 2}(:, 1),...
        eye_tracking_data_pca{person_idx}{exp1_idx_pca(person_idx) + (exp_plot_idx-1), 2}(:, 2), 'o')
    ax = gca();
    ax_min = min(ax.XLim(1), ax.YLim(1));
    ax_max = max(ax.XLim(2), ax.YLim(2));
    ax.XLim = [ax_min, ax_max];
    ax.YLim = [ax_min, ax_max];
end