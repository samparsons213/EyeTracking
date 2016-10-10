% Loads all the patients and parses all the trials
clear all;
close all;

folder = './data/';
files = dir(folder);

count = 1;
for f=1:length(files),
    file = files(f);
    if(regexp(file.name, '\.asc', 'ONCE'))
        patients{count} = preprocessPatient(strcat(folder, file.name));
        count = count+1;
    end
end

fprintf('Processed %d patients:\n', count);
for i=1:length(patients),
fprintf('%s\n', patients{i}.name);
end

% Save to file
fprintf('Saving data\n');
save('parsedPatients.mat', 'patients');