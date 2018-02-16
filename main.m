clc;
clear all;
close all;

% names = [{'dermatology'}; {'wine'}; {'glass'}; {'sonar'}; {'ionosphere'};...
%     {'spectfheart'}; {'bupa'}; {'wdbc'}; {'haberman'};{'pima'};...
%     {'australian'}; {'saheart'}; {'satimage'}; {'segment'}];

% names = [{'short_HGV'}; {'medium_HGV'}; {'long_HGV'}; {'nott_bc'}];

% names = [{'dermatology'}; {'wine'}; {'glass'}; {'sonar'};...
%     {'ionosphere'}; {'spectfheart'}; {'bupa'}; {'wdbc'}; {'haberman'};...
%     {'pima'}; {'australian'}; {'saheart'}; {'satimage'}; {'segment'}; {'ecoli'}];
names = [{'mammographic'};{'credit-approval'};{'ozone'};...
    {'tic-tac-toe'};{'ilpd'}];

files = [{'l1_accuracy.xlsx'}; {'l1_auc.xlsx'}; {'l1_sensitivity.xlsx'};...
    {'l1_specificity.xlsx'}; {'l1_time.xlsx'}];
% columns = [{'A2'}; {'B2'}; {'C2'};{'D2'}; {'E2'}; {'F2'}; {'G2'}; {'H2'}; {'I2'}; {'J2'}; {'K2'}; {'L2'}; {'M2'}; {'N2'}];
columns = [{'C2'};{'D2'}; {'E2'}; {'F2'}; {'G2'}];

ntimes = 2;
for loop = 1:size(names,1)
%     cd 'E:\dropbox\Dropbox\projects\phd\codes\matlab\fm_fi\datasets'
    cd 'C:\Users\utkarsh\Dropbox\projects\phd\codes\matlab\fm_fi\datasets'
    [X_orig, y_orig] = datasets1(names{loop}, 1);
%     cd 'E:\dropbox\Dropbox\projects\phd\codes\matlab\fm_fi\ALGO together comparison'
    cd 'C:\Users\utkarsh\Dropbox\projects\phd\codes\matlab\fm_fi\ALGO together comparison'
    
    params = ensemble_classification_common(X_orig, y_orig);
    
    for i = 1:length(files)
        xlswrite(files{i}, params(:,i), 'sheet1', columns{loop});
    end
    
end