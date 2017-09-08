# (c) 2017 Marian Longa
# v6: change holdout validation method to k-fold cross-validation

import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
import re
import datetime
import sys

# import data from TSV file
data = pd.read_csv('dataset_new_combined_20170804.tsv', sep='\t')
data = data.dropna(subset = ['is_fake_news_2'])
data = data.drop(data[data.is_fake_news_2 == 'UNKNOWN'].index)
data['fake'] = (data.is_fake_news_2 == 'TRUE').astype(int) # <-- NB changed FALSE to TRUE in this version ==> positive correlation corresponds to fake news not real news!
data['user_verified'] = data['user_verified'].astype(int)

# add derived features related to various base features


# derived feature functions
def number_of_swears(text):
    number = 0
    swearwords = ['fuck', 'shit', 'dumb', 'retard', 'kill', 'crap']
    for swearword in swearwords:
        number += len(re.findall(swearword, text.lower()))
    return number
def extract_weekday_number(text):
    weekdays = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}
    weekday = re.search('^([A-Z][a-z][a-z])\s', text).group(1)
    return weekdays[weekday]
def get_time_delta(datetime_created_string):
    datetime_today = datetime.datetime.now()
    datetime_created = datetime.datetime.strptime(datetime_created_string, "%a %b %d %H:%M:%S +0000 %Y")
    datetime_delta = datetime_today - datetime_created
    return datetime_delta.days
#def get_user_screen_name_num_caps_digits_except_first_cap(text):
#    return len(re.findall('[A-Z0-9]', text)) - len(re.findall('^[A-Z]', text))
#def get_user_screen_name_has_caps_digits_except_first_cap(text):
#    return (len(re.findall('[A-Z0-9]', text)) - len(re.findall('^[A-Z]', text))) >= 1

# 'user_screen_name' related features
data['user_screen_name_has_caps'] = data['user_screen_name'].apply(lambda text: int(len(re.findall('[A-Z]', text)) >= 1))
data['user_screen_name_has_digits'] = data['user_screen_name'].apply(lambda text: int(len(re.findall('[0-9]', text)) >= 1))
data['user_screen_name_has_underscores'] = data['user_screen_name'].apply(lambda text: int(len(re.findall('[_]', text)) >= 1))
data['user_screen_name_has_caps_digits'] = data['user_screen_name'].apply(lambda text: int(len(re.findall('[A-Z0-9]', text)) >= 1))
data['user_screen_name_has_caps_underscores'] = data['user_screen_name'].apply(lambda text: int(len(re.findall('[A-Z_]', text)) >= 1))
data['user_screen_name_has_digits_underscores'] = data['user_screen_name'].apply(lambda text: int(len(re.findall('[0-9_]', text)) >= 1))
data['user_screen_name_has_caps_digits_underscores'] = data['user_screen_name'].apply(lambda text: int(len(re.findall('[A-Z0-9_]', text)) >= 1))
data['user_screen_name_num_caps'] = data['user_screen_name'].apply(lambda text: len(re.findall('[A-Z]', text)))
data['user_screen_name_num_digits'] = data['user_screen_name'].apply(lambda text: len(re.findall('[0-9]', text)))
data['user_screen_name_num_underscores'] = data['user_screen_name'].apply(lambda text: len(re.findall('[_]', text)))
data['user_screen_name_num_caps_digits'] = data['user_screen_name'].apply(lambda text: len(re.findall('[A-Z0-9]', text)))
data['user_screen_name_num_caps_underscores'] = data['user_screen_name'].apply(lambda text: len(re.findall('[A-Z_]', text)))
data['user_screen_name_num_digits_underscores'] = data['user_screen_name'].apply(lambda text: len(re.findall('[0-9_]', text)) )
data['user_screen_name_num_caps_digits_underscores'] = data['user_screen_name'].apply(lambda text: len(re.findall('[A-Z0-9_]', text)))
#data['user_screen_name_has_caps_digits_except_first_cap'] = data['user_screen_name'].apply(get_user_screen_name_has_caps_digits_except_first_cap)
#data['user_screen_name_num_caps_digits_except_first_cap'] = data['user_screen_name'].apply(get_user_screen_name_num_caps_digits_except_first_cap)
data['user_screen_name_has_weird_chars'] = data['user_screen_name'].apply(lambda text: int(len(re.findall('[^A-Za-z .\']', text)) >= 1))
data['user_screen_name_num_weird_chars'] = data['user_screen_name'].apply(lambda text: len(re.findall('[^A-Za-z .\']', text)))

# 'text' related features
data['text_num_caps'] = data['text'].apply(lambda text: len(re.findall('[A-Z]', text)))
data['text_num_digits'] = data['text'].apply(lambda text: len(re.findall('[0-9]', text)))
data['text_num_nonstandard'] = data['text'].apply(lambda text: len(re.findall('[^A-Za-z0-9,.]', text)))
data['text_num_nonstandard_extended'] = data['text'].apply(lambda text: len(re.findall('[^A-Za-z0-9,.?!\-@#\']', text)))
data['text_num_exclam'] = data['text'].apply(lambda text: len(re.findall('[!]', text)))
data['text_num_caps_exclam'] = data['text'].apply(lambda text: len(re.findall('[A-Z!]', text)))
data['text_num_caps_digits'] = data['text'].apply(lambda text: len(re.findall('[A-Z0-9]', text)))
data['text_num_caps_digits_exclam'] = data['text'].apply(lambda text: len(re.findall('[A-Z0-9!]', text)))
data['text_num_swears'] = data['text'].apply(number_of_swears)

# 'created_at' related features
data['created_at_hour'] = data['created_at'].apply(lambda text: int(re.search('\s([0-9][0-9]):', text).group(1)))
data['created_at_hour_23_to_5'] = data['created_at_hour'].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)
data['created_at_hour_13_to_22'] = data['created_at_hour'].isin(range(13, 23)).astype(int)
data['created_at_weekday'] = data['created_at'].apply(extract_weekday_number)
data['created_at_weekday_sun_mon_tue'] = data['created_at_weekday'].isin([7, 1, 2]).astype(int)

# 'user_description' related features
data['user_description_num_caps'] = data['user_description'].apply(lambda text: len(re.findall('[A-Z]', text)))
data['user_description_num_digits'] = data['user_description'].apply(lambda text: len(re.findall('[0-9]', text)))
data['user_description_num_nonstandard'] = data['user_description'].apply(lambda text: len(re.findall('[^A-Za-z0-9,.]', text)))
data['user_description_num_nonstandard_extended'] = data['user_description'].apply(lambda text: len(re.findall('[^A-Za-z0-9,.?!\-@#\']', text)))
data['user_description_num_exclam'] = data['user_description'].apply(lambda text: len(re.findall('[!]', text)))
data['user_description_num_caps_with_num_nonstandard'] = data['user_description'].apply(lambda text: len(re.findall('[^a-z0-9,.]', text)))
data['user_description_num_non_a_to_z'] = data['user_description'].apply(lambda text: len(re.findall('[^a-z]', text)))
data['user_description_num_non_a_to_z_non_digits'] = data['user_description'].apply(lambda text: len(re.findall('[^a-z0-9]', text)))
data['user_description_num_caps_exclam'] = data['user_description'].apply(lambda text: len(re.findall('[A-Z!]', text)))

# 'user_name' related features
data['user_name_has_caps'] = data['user_name'].apply(lambda text: int(len(re.findall('[A-Z]', text)) >= 1))
data['user_name_has_digits'] = data['user_name'].apply(lambda text: int(len(re.findall('[0-9]', text)) >= 1))
data['user_name_has_underscores'] = data['user_name'].apply(lambda text: int(len(re.findall('[_]', text)) >= 1))
data['user_name_has_caps_digits'] = data['user_name'].apply(lambda text: int(len(re.findall('[A-Z0-9]', text)) >= 1))
data['user_name_has_caps_underscores'] = data['user_name'].apply(lambda text: int(len(re.findall('[A-Z_]', text)) >= 1))
data['user_name_has_digits_underscores'] = data['user_name'].apply(lambda text: int(len(re.findall('[0-9_]', text)) >= 1))
data['user_name_has_caps_digits_underscores'] = data['user_name'].apply(lambda text: int(len(re.findall('[A-Z0-9_]', text)) >= 1))
data['user_name_num_caps'] = data['user_name'].apply(lambda text: len(re.findall('[A-Z]', text)))
data['user_name_num_digits'] = data['user_name'].apply(lambda text: len(re.findall('[0-9]', text)))
data['user_name_num_underscores'] = data['user_name'].apply(lambda text: len(re.findall('[_]', text)))
data['user_name_num_caps_digits'] = data['user_name'].apply(lambda text: len(re.findall('[A-Z0-9]', text)))
data['user_name_num_caps_underscores'] = data['user_name'].apply(lambda text: len(re.findall('[A-Z_]', text)))
data['user_name_num_digits_underscores'] = data['user_name'].apply(lambda text: len(re.findall('[0-9_]', text)) )
data['user_name_num_caps_digits_underscores'] = data['user_name'].apply(lambda text: len(re.findall('[A-Z0-9_]', text)))
data['user_name_has_weird_chars'] = data['user_name'].apply(lambda text: int(len(re.findall('[^A-Za-z .\']', text)) >= 1))
data['user_name_num_weird_chars'] = data['user_name'].apply(lambda text: len(re.findall('[^A-Za-z .\']', text)))
data['user_name_has_nonprintable_chars'] = data['user_name'].apply(lambda text: int(len(re.findall('[^ -~]', text)) >= 1))
data['user_name_num_nonprintable_chars'] = data['user_name'].apply(lambda text: len(re.findall('[^ -~]', text)) >= 1)

# nonzero number features
data['num_urls_is_nonzero'] = data['num_urls'].apply(lambda number: int(number >= 1))
data['num_media_is_nonzero'] = data['num_media'].apply(lambda number: int(number >= 1))
data['num_hashtags_is_nonzero'] = data['num_hashtags'].apply(lambda number: int(number >= 1))
data['num_mentions_is_more_than_2'] = data['num_hashtags'].apply(lambda number: int(number > 2))

# per-unit-time related features
data['user_created_at_delta'] = data['user_created_at'].apply(get_time_delta) # number of days from account creation to now
data['created_at_delta'] = data['user_created_at'].apply(get_time_delta) # number of days from tweet creation to now
data['user_statuses_count_per_day'] = data['user_statuses_count'] / data['user_created_at_delta']
data['user_followers_count_per_day'] = data['user_followers_count'] / data['user_created_at_delta']
data['user_listed_count_per_day'] = data['user_listed_count'] / data['user_created_at_delta']
data['user_friends_count_per_day'] = data['user_friends_count'] / data['user_created_at_delta']
data['user_favourites_count_per_day'] = data['user_favourites_count'] / data['user_created_at_delta']
data['retweet_count_per_day'] = data['retweet_count'] / data['created_at_delta']

# need-to-convert-format features
data['user_default_profile'] = data['user_default_profile'].astype(int)
data['user_profile_use_background_image'] = data['user_profile_use_background_image'].astype(int)
data['user_default_profile_image'] = data['user_default_profile_image'].astype(int)


# select relevant features based on their quality (see 'features to use'.docx file)
features_basic_all = \
    'fake ~ retweet_count + user_verified + user_friends_count + user_followers_count + user_favourites_count + ' \
    'num_hashtags + num_mentions + num_urls + num_media'
features_basic_some = \
    'fake ~ user_verified + user_friends_count + user_followers_count + num_urls + num_media'
features_basic_few = \
    'fake ~ user_verified + user_followers_count + num_urls'
features_extended_few_single = \
    'fake ~ user_verified + text_num_caps_digits + user_screen_name_has_caps_digits + num_urls_is_nonzero + ' \
    'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
    'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
    'user_name_has_weird_chars + user_default_profile'
features_extended_few_multiple = \
    'fake ~ user_verified + text_num_caps + text_num_digits + user_screen_name_has_caps + user_screen_name_has_digits + num_urls_is_nonzero + ' \
    'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
    'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
    'user_name_has_weird_chars + user_default_profile'
features_extended_some_single = \
    'fake ~ user_verified + text_num_caps_digits + user_screen_name_has_caps_digits + num_urls_is_nonzero + ' \
    'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
    'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
    'user_name_has_weird_chars + user_default_profile + ' \
    'created_at_weekday_sun_mon_tue + created_at_hour_13_to_22 + user_friends_count_per_day + num_media + ' \
    'created_at_hour_23_to_5 + text_num_swears + user_profile_use_background_image + created_at_weekday + ' \
    'user_listed_count + created_at_hour + user_friends_count + user_created_at_delta + user_statuses_count'
features_extended_some_single_without_biasing_features = \
    'fake ~ user_verified + text_num_caps_digits + user_screen_name_has_caps_digits + num_urls_is_nonzero + ' \
    'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
    'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
    'user_name_has_weird_chars + user_default_profile + ' \
    'created_at_weekday_sun_mon_tue + user_friends_count_per_day + num_media + ' \
    'user_profile_use_background_image + created_at_weekday + ' \
    'user_listed_count + created_at_hour + user_friends_count + user_created_at_delta + user_statuses_count'
features_extended_some_multiple = \
    'fake ~ user_verified + text_num_caps + text_num_digits + user_screen_name_has_caps + user_screen_name_has_digits + num_urls_is_nonzero + ' \
    'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
    'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
    'user_name_has_weird_chars + user_default_profile + ' \
    'created_at_weekday_sun_mon_tue + created_at_hour_13_to_22 + user_friends_count_per_day + num_media + ' \
    'created_at_hour_23_to_5 + text_num_swears + user_profile_use_background_image + created_at_weekday + ' \
    'user_listed_count + created_at_hour + user_friends_count + user_created_at_delta + user_statuses_count'
features_extended_some_multiple_without_biasing_features = \
    'fake ~ user_verified + text_num_caps + text_num_digits + user_screen_name_has_caps + user_screen_name_has_digits + num_urls_is_nonzero + ' \
    'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
    'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
    'user_name_has_weird_chars + user_default_profile + ' \
    'created_at_weekday_sun_mon_tue + user_friends_count_per_day + num_media + ' \
    'user_profile_use_background_image + created_at_weekday + ' \
    'user_listed_count + created_at_hour + user_friends_count + user_created_at_delta + user_statuses_count'

# choose feature set for training classifier
features = features_extended_some_multiple

y, X = dmatrices(features, data, return_type='dataframe')
y_array, X_array = np.ravel(y.values), X.values

# use a Stratified K-Fold cross-validation method with minority class upsampling during training
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
binary_scores = []
probabilistic_scores = []
confusion_matrices = []
classification_reports = []
weights_list = []
for train_index, test_index in cv.split(X_array, y_array):
    # select train and test data based on indices from the Stratified K-Fold Cross-Validation function
    X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index,:], y.iloc[train_index,:], y.iloc[test_index,:]
    data_train = pd.concat([y_train, X_train], axis=1)

    # in the training set, upsample minority class (ie fake news class) so that it has the same number of rows as the majority class (ie real news class)
    data_train_real = data_train[data_train.fake == 0]  # print "data_real shape: ", data_real.shape[0]
    data_train_fake = data_train[data_train.fake == 1]  # print "data_fake shape: ", data_fake.shape[0]
    data_train_fake_upsampled = resample(data_train_fake, replace=True, n_samples=data_train_real.shape[0], random_state=123)
    data_train_upsampled = pd.concat([data_train_real, data_train_fake_upsampled])
    y_train_upsampled, X_train_upsampled = dmatrices(features, data_train_upsampled, return_type='dataframe')

    # convert pandas dataframes into numpy arrays
    y_train_upsampled_array, X_train_upsampled_array = np.ravel(y_train_upsampled.values), X_train_upsampled.values
    y_test_array, X_test_array = np.ravel(y_test.values), X_test.values

    # fit model
    model = LogisticRegression(C=1e10, solver='newton-cg', max_iter=10000)  # newton-cg is the only solver which seems to work with this data set
    model.fit(X_train_upsampled_array, y_train_upsampled_array)

    # calculate statistics for one fold
    binary_scores.append(metrics.accuracy_score(y_test_array, model.predict(X_test_array)))
    probabilistic_scores.append(metrics.roc_auc_score(y_test_array, model.predict_proba(X_test_array)[:, 1]))
    confusion_matrices.append(metrics.confusion_matrix(y_test_array, model.predict(X_test_array)))
    classification_reports.append(metrics.classification_report(y_test, model.predict(X_test)))

    # store feature weights (coefficients)
    #formatted_coeffs = ['{:+0.15f}'.format(coeff) for coeff in (model.coef_)[0]]
    #weights_list = [X.columns.tolist(), np.transpose(formatted_coeffs).tolist()]
    #weights_list = map(list, zip(*weights_list))
    #weights_list_sorted = sorted(weights_list, key=lambda column: abs(float(column[1])), reverse=True)
    #feature_weights.append(weights_list_sorted)
    #weights = [coeff for coeff in (model.coef_)[0]]
    #weights_list = np.transpose(formatted_coeffs).tolist()
    #weights_list = map(list, zip(*weights_list))
    #weights_sorted = sorted(weights, reverse=True)
    #weights_list.append(weights_list_sorted)
    weights_list.append([coeff for coeff in (model.coef_)[0]])

# print statistics averaged over all folds
#print("binary scores: ", binary_scores)
print("mean binary score: ", np.mean(binary_scores))
#print("probabilistic scores: ", probabilistic_scores)
print("mean probabilistic score: ", np.mean(probabilistic_scores))
#print("confusion matrices: ", confusion_matrices)
print("mean confusion matrix: ", np.mean(confusion_matrices, axis=0))
#print("weights list: ", weights_list)
weights_mean = np.mean(weights_list, axis=0)
features_weights = [X.columns.tolist(), weights_mean.tolist()]
features_weights = map(list, zip(*features_weights))
features_weights_sorted = sorted(features_weights, key=lambda column: abs(column[1]), reverse=True)
print("mean feature weights: ", features_weights_sorted)
