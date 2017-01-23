function sorted_fnames = sortedFilenames(sorted_list, ctrl_fnames,...
    ad_fnames, pca_fnames)
% Takes a sorted list of patients, labelled by position in their particular
% filenames list and with a label identifying which list they belong to,
% and returns the corresponding sorted list of filenames

% Inputs:

% sorted_list:      n_patients by 2 integer array, with first column giving
%                   position within their filename list, and second column
%                   a label in [1-3] identifying which list they belong to

% ctrl_fnames:      n_ctrl_patients by 1 cell array, with each element
%                   giving the filename of one patient in the control group

% ad_fnames:        n_ad_patients by 1 cell array, with each element
%                   giving the filename of one patient in the ad group

% pca_fnames:       n_ctrl_patients by 1 cell array, with each element
%                   giving the filename of one patient in the ad group

% Outputs:

% sorted_fnames:    n_patients by 1 cell array, with each element giving
%                   the filename of patient in the correpsonding position
%                   in sorted_list

% Author:           Sam Parsons
% Date created:     18/10/2016
% Last amended:     18/10/2016

%     *********************************************************************
%     Check inputs:
%     All 4 args must be input
    if nargin < 4
        error('all 4 args must be input')
    end
%     sorted_list must be an [n 2] integer array for some positive n >=3
%     (there is assumed to be at least one patient from each group)
    s_list = size(sorted_list);
    if ~(isnumeric(sorted_list) && isreal(sorted_list) &&...
            ismatrix(sorted_list) && (s_list(1) >= 3) && (s_list(2) == 2) &&...
            all(sorted_list(:) == round(sorted_list(:))))
        err_msg = ['sorted_list must be an [n 2] integer array for some',...
            ' positive n >=3'];
        error(err_msg)
    end
%     ctrl_fnames must be a column vector cell array of strings
    if ~(iscolumn(ctrl_fnames) && iscellstr(ctrl_fnames))
        error('ctrl_fnames must be a column vector cell array of strings')
    end
%     ad_fnames must be a column vector cell array of strings
    if ~(iscolumn(ad_fnames) && iscellstr(ad_fnames))
        error('ad_fnames must be a column vector cell array of strings')
    end
%     pca_fnames must be a column vector cell array of strings
    if ~(iscolumn(pca_fnames) && iscellstr(pca_fnames))
        error('pca_fnames must be a column vector cell array of strings')
    end
%     All entries in sorted_list must exist in corresponding fnames cell
%     array
%     Control patients
    if min(sorted_list(:, 1)) < 1
        error('all entries in col 1 of sorted_list must be >= 1')
    end
    n_ctrl_patients = length(ctrl_fnames);
    ctrl_patients_idx = sorted_list(:, 2) == 1;
    if any(sorted_list(ctrl_patients_idx, 1) > n_ctrl_patients)
        error('All control patients in sorted list must have a filename')
    end
%     AD patients
    n_ad_patients = length(ad_fnames);
    ad_patients_idx = sorted_list(:, 2) == 2;
    if any(sorted_list(ad_patients_idx, 1) > n_ad_patients)
        error('All AD patients in sorted list must have a filename')
    end
%     PCA patients
    n_pca_patients = length(pca_fnames);
    pca_patients_idx = sorted_list(:, 2) == 3;
    if any(sorted_list(pca_patients_idx, 1) > n_pca_patients)
        error('All PCA patients in sorted list must have a filename')
    end
%     *********************************************************************

%     *********************************************************************
%     Main body of function. For each row of sorted_list, extract relevant
%     name from fnames cell array corresponding to its col 2 value.
%     *********************************************************************

    sorted_fnames = cell(s_list(1), 1);
    for row_idx = 1:s_list(1)
        switch sorted_list(row_idx, 2)
            case 1
                sorted_fnames{row_idx} =...
                    ctrl_fnames{sorted_list(row_idx, 1)};
            case 2
                sorted_fnames{row_idx} =...
                    ad_fnames{sorted_list(row_idx, 1)};
            case 3
                sorted_fnames{row_idx} =...
                    pca_fnames{sorted_list(row_idx, 1)};
        end
    end

end