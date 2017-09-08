data = xlsread('../model_performances/all/svm-performances-20170821-for-matlab.xlsx', 'A1:Q5500');

kernel        = data(:, 3);
log_max_iter  = data(:, 5);
poly_degree   = data(:, 6);
log_c         = data(:, 8);
roc_auc_score = data(:, 10);
f1_score      = data(:, 13);

figure
%scatter3(log_max_iter, log_c, roc_auc_score)
%scatter(log_c, roc_auc_score)
scatter(log_max_iter, roc_auc_score)