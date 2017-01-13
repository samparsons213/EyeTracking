% Script for saving plots of each experiment for each person to file
load('original_data.mat')
n_experiments = 12;
n_ctrl = 20;
n_ad = 24;
n_pca = 6;
exp1_idx = arrayfun(@(person_idx) size(eye_tracking_data{person_idx}, 1), 1:n_ctrl)-n_experiments+1;
exp1_idx_ad = arrayfun(@(person_idx) size(eye_tracking_data_ad{person_idx}, 1), 1:n_ad)-n_experiments+1;
exp1_idx_pca = arrayfun(@(person_idx) size(eye_tracking_data_pca{person_idx}, 1), 1:n_pca)-n_experiments+1;
folder_root = 'C:\Users\Sam\OneDrive\C-PLACID\Latex\Eye Tracking\EyeTrackingPlots\';
file_names = {'armgr.asc'; 'biged.asc'; 'briel.asc'; 'conma.asc';...
    'frani.asc'; 'golsh.asc'; 'grejo.asc'; 'holtr.asc'; 'lewwi.asc';...
    'manjo.asc'; 'mckro.asc'; 'medca.asc'; 'newal.asc'; 'pagjo.asc';...
    'pleri.asc'; 'redjo.asc'; 'strmi.asc'; 'staro.asc'; 'stegr.asc';...
    'towsu.asc'};
ad_file_names = {'baria.asc'; 'bigba.asc'; 'bunda.asc'; 'golro.asc';...
    'harja.asc'; 'hayma.asc'; 'jammi.asc'; 'keeke.asc'; 'kinne.asc';...
    'lewhe.asc'; 'macma.asc'; 'medst.asc'; 'newsi.asc'; 'pagje.asc';...
    'redel.asc'; 'selro.asc'; 'shake.asc'; 'sheca.asc'; 'shejo.asc';...
    'stewe.asc'; 'thoan.asc'; 'towro.asc'; 'warma.asc'; 'wilfa.asc'};
pca_file_names = {'conpa.asc'; 'engsa.asc'; 'frara.asc'; 'jarth.asc';...
    'morpe.asc'; 'staph.asc'};

for exp_idx = 1:n_experiments
    folder_name = [folder_root, 'Exp_', num2str(exp_idx), '\'];
    for person_idx = 1:n_ctrl
        plot(eye_tracking_data{person_idx}{exp1_idx(person_idx)+exp_idx-1, 2}, 'o')
        title(file_names{person_idx}(1:5))
        xlabel(['Exp: ', num2str(exp_idx)])
        ax = gca();
        ax.XLim = [0 3000];
        saveas(gcf, [folder_name, file_names{person_idx}(1:5), '.jpg'])
        close(gcf)
    end
    for person_idx = 1:n_ad
        plot(eye_tracking_data_ad{person_idx}{exp1_idx_ad(person_idx)+exp_idx-1, 2}, 'o')
        title(ad_file_names{person_idx}(1:5))
        xlabel(['Exp: ', num2str(exp_idx)])
        ax = gca();
        ax.XLim = [0 3000];
        saveas(gcf, [folder_name, ad_file_names{person_idx}(1:5), '.jpg'])
        close(gcf)
    end
    for person_idx = 1:n_pca
        plot(eye_tracking_data_pca{person_idx}{exp1_idx_pca(person_idx)+exp_idx-1, 2}, 'o')
        title(pca_file_names{person_idx}(1:5))
        xlabel(['Exp: ', num2str(exp_idx)])
        ax = gca();
        ax.XLim = [0 3000];
        saveas(gcf, [folder_name, pca_file_names{person_idx}(1:5), '.jpg'])
        close(gcf)
    end
end