function [eye_tracking_data, target_data] = loadManyFiles(file_folders,...
    file_names)
% Loads the eye tracking and target for multiple persons' files. The number
% of experiments in each file is assumed to be constant

% Inputs:

% file_folders:     either a character string giving the folder that all
%                   data files are in, or an n_files by 1 cell array of
%                   character strings, each giving the folder that each
%                   file is in

% file_names:       n_files by 1 cell array of character strings, each
%                   giving the name of one data file

% Outputs:

% eye_tracking_data:if only one file_name is input then output is just as
%                   per eye_tracking_data output from loadOneFile.m. If
%                   more than one file_name input then n_files by 1 cell
%                   array is output, each element of which is a n by 2 cell
%                   array as per eye_tracking_data output from
%                   loadOneFile.m

% target_data:      if only one file_name is input then output is just as
%                   per tracking_data output from loadOneFile.m. If more
%                   than one file_name input then n_files by 1 cell array
%                   is output, each element of which is a n by 2 cell
%                   array as per tracking_data output from loadOneFile.m

% Author:           Sam Parsons
% Date created:     27/09/2016
% Last amended:     27/09/2016

%     *********************************************************************
%     Check inputs
%     *********************************************************************
    if nargin < 2
        error('Both arguments must be input')
    end
%     file_folders must be either a character string or an n_files by 1
%     cell array of character strings
    if ~((iscellstr(file_folders) || ischar(file_folders)) &&...
            ~isempty(file_folders))
        err_msg = ['file_folders must either be a character string or an',...
            ' n_files by 1 cell array of character stings'];
        error(err_msg)
    end
    if iscellstr(file_folders) && (length(file_folders) > 1)
        file_folders = file_folders(:);
        n_files = length(file_folders);
    else
        n_files = length(file_names(:));
        if ischar(file_folders)
            file_folders = {file_folders};
        end
    end
%     file_names must be an n_files by 1 cell array of character strings
    file_names = file_names(:);
    if ~(iscellstr(file_names) && (length(file_names) == n_files))
        err_msg = ['file_names must be an n_files by 1 cell array of',...
            ' character strings'];
        error(err_msg)
    end
%     *********************************************************************
%     Main body of code. If n_files = 1 then procedure just runs
%     loadOneFile.m for the file. If n_files > 1, then loadOneFile.m is run
%     on each file in turn
%     *********************************************************************

    switch n_files
        case 1
            [eye_tracking_data, target_data] = loadOneFile(file_folders{1},...
                file_names{1});
        otherwise
            eye_tracking_data = cell(n_files, 1);
            target_data = cell(n_files, 1);
            for file_idx = 1:n_files
                folder_name = file_folders{min(length(file_folders), file_idx)};
                [eye_tracking_data{file_idx}, target_data{file_idx}] =...
                    loadOneFile(folder_name, file_names{file_idx});
            end
    end

end