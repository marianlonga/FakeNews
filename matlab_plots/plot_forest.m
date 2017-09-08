%data = xlsread('../model_performances/all/forest-performances-20170908-for-matlab.xlsx', 'A1:O255002');

n_estimators      = data(:, 3);
max_depth         = data(:, 4);
min_samples       = data(:, 5);
max_features_type = data(:, 6);
roc_auc_score     = data(:, 8);

figure
%scatter3(n_estimators, max_depth, roc_auc_score)
%scatter(n_estimators, roc_auc_score)
%scatter(max_depth, roc_auc_score)
%scatter(min_samples, roc_auc_score)
%scatter3(n_estimators, min_samples, roc_auc_score)
scatter3(n_estimators, max_features_type, roc_auc_score)