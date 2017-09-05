# (c) 2017 Marian Longa

# TODO: try LinearSVM instead of SVM.SVC --> improvement?
# TODO: try normalising inputs to SVM to [-1,+1] --> convergence?, improvement?
# TODO: try looping through gamma parameter for RBF kernel in SVM
# TODO: for logistic regression use features selected using the new method (where small percentage difference mean is eliminated)

import pandas as pd
from patsy import dmatrices
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
import re
import datetime
import csv
import multiprocessing as mp
import warnings

PERFORMANCES_FILE_NAME  = 'performances.csv'                                                   # default 'performances.csv'
MODEL_NAME              = 'log_reg'                                                            # choose 'log_reg', 'svm', 'knn'
CV_NUMBER_OF_SPLITS     = 5                                                                    # number of splits for cross-validation, default = 5
USE_SCALED_DATA         = True                                                                 # default True
SVM_FEATURES_TO_USE     = 'features_extended_some_multiple_without_text_num_swears'            # default 'features_extended_some_multiple_without_text_num_swears'
SVM_NUMBER_OF_PROCESSES = 47                                                                   # number of processes to use for parallel computing in SVM, default 4 (47 for DSI cluster)
SVM_KERNEL_LIST         = ['linear', 'poly', 'rbf', 'sigmoid']                                 # SVM kernel list, default ['linear', 'poly', 'rbf', 'sigmoid']
SVM_MAX_ITER_LIST       = [(k * (10 ** exp)) for exp in range(1, 7, 1) for k in [1, 5]]        # SVM max iter list, default [10, 100, 1000, 5000, 10000, 50000, 100000]
SVM_C_LIST              = [(k * (10 ** exp)) for exp in range(-15, 16, 1) for k in [1, 5]]     # SVM C list, default [(k * (10 ** exp)) for exp in range(-10, 11, 1) for k in [1, 5]]
SVM_CACHE_SIZE          = 5000                                                                 # default 200 (5000 for DSI cluster)
SVM_POLY_DEGREE_LIST    = [2, 3, 4, 5]                                                         # default [2,3,4,5]
KNN_FEATURES_TO_USE     = 'features_extended_some_multiple_without_text_num_swears'            # default 'features_extended_some_multiple_without_text_num_swears'
KNN_N_NEIGHBORS_LIST    = range(1, 201)                                                        # KNN number of nearest neighbors to use, default range(1, 51)
RESAMPLE                = 'up'                                                                 # whether to upsample minority class ('up' -- default) or downsample majority class ('down') or use the original data distribution ('none')

# define all features
statistics_features = [
    'tweet_id', 'retweet_count', 'user_verified', 'user_friends_count', 'user_followers_count',
    'user_favourites_count', 'geo_coordinates', 'num_hashtags', 'num_mentions', 'num_urls',
    'num_media', 'num_media_is_nonzero',
    'text_num_caps', 'text_num_digits', 'text_num_nonstandard', 'text_num_nonstandard_extended',
    'text_num_exclam', 'text_num_caps_exclam', 'text_num_caps_digits', 'text_num_caps_digits_exclam',
    'text_num_swears',
    'num_urls_is_nonzero', 'num_hashtags_is_nonzero', 'num_mentions_is_more_than_2',
    'created_at_hour', 'created_at_hour_08_to_17', 'created_at_hour_18_to_00', 'created_at_weekday', 'created_at_weekday_sun_mon_tue',
    'created_at_hour_of_week',
    'user_description_num_caps', 'user_description_num_digits', 'user_description_num_nonstandard',
    'user_description_num_nonstandard_extended', 'user_description_num_exclam',
    'user_description_num_caps_with_num_nonstandard', 'user_description_num_non_a_to_z',
    'user_description_num_non_a_to_z_non_digits', 'user_description_num_caps_exclam',
    'user_default_profile_image',
    'user_listed_count',
    'user_profile_use_background_image', 'user_default_profile',
    'user_screen_name_has_caps', 'user_screen_name_has_digits', 'user_screen_name_has_underscores',
    'user_screen_name_has_caps_digits', 'user_screen_name_has_caps_underscores', 'user_screen_name_has_digits_underscores',
    'user_screen_name_has_caps_digits_underscores', 'user_screen_name_num_caps', 'user_screen_name_num_digits',
    'user_screen_name_num_underscores', 'user_screen_name_num_caps_digits', 'user_screen_name_num_caps_underscores',
    'user_screen_name_num_digits_underscores', 'user_screen_name_num_caps_digits_underscores',
    'user_screen_name_has_weird_chars', 'user_screen_name_num_weird_chars',
    'user_name_has_caps', 'user_name_has_digits', 'user_name_has_underscores', 'user_name_has_caps_digits',
    'user_name_has_caps_underscores', 'user_name_has_digits_underscores', 'user_name_has_caps_digits_underscores',
    'user_name_num_caps', 'user_name_num_digits', 'user_name_num_underscores', 'user_name_num_caps_digits',
    'user_name_num_caps_underscores', 'user_name_num_digits_underscores', 'user_name_num_caps_digits_underscores',
    'user_name_has_weird_chars', 'user_name_num_weird_chars', 'user_name_has_nonprintable_chars',
    'user_name_num_nonprintable_chars',
    'user_statuses_count', 'user_created_at_delta', 'user_statuses_count_per_day', 'user_followers_count_per_day',
    'user_listed_count_per_day', 'user_friends_count_per_day', 'user_favourites_count_per_day', 'retweet_count_per_day'
]

# define relevant features based on their quality (see 'features to use'.docx file)
FEATURES = {
    'features_basic_all':
        'fake ~ retweet_count + user_verified + user_friends_count + user_followers_count + user_favourites_count + ' \
        'num_hashtags + num_mentions + num_urls + num_media',
    'features_basic_all':
        'fake ~ retweet_count + user_verified + user_friends_count + user_followers_count + user_favourites_count + ' \
        'num_hashtags + num_mentions + num_urls + num_media',
    'features_basic_some':
        'fake ~ user_verified + user_friends_count + user_followers_count + num_urls + num_media',
    'features_basic_few':
        'fake ~ user_verified + user_followers_count + num_urls',
    'features_extended_few_single':
        'fake ~ user_verified + text_num_caps_digits + user_screen_name_has_caps_digits + num_urls_is_nonzero + ' \
        'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
        'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
        'user_name_has_weird_chars + user_default_profile',
    'features_extended_few_multiple':
        'fake ~ user_verified + text_num_caps + text_num_digits + user_screen_name_has_caps + user_screen_name_has_digits + num_urls_is_nonzero + ' \
        'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
        'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
        'user_name_has_weird_chars + user_default_profile',
    'features_extended_some_single':
        'fake ~ user_verified + text_num_caps_digits + user_screen_name_has_caps_digits + num_urls_is_nonzero + ' \
        'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
        'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
        'user_name_has_weird_chars + user_default_profile + ' \
        'created_at_weekday_sun_mon_tue + created_at_hour_08_to_17 + user_friends_count_per_day + num_media + ' \
        'created_at_hour_18_to_00 + text_num_swears + user_profile_use_background_image + created_at_weekday + ' \
        'user_listed_count + created_at_hour + user_friends_count + user_created_at_delta + user_statuses_count',
    'features_extended_some_single_without_biasing_features':
        'fake ~ user_verified + text_num_caps_digits + user_screen_name_has_caps_digits + num_urls_is_nonzero + ' \
        'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
        'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
        'user_name_has_weird_chars + user_default_profile + ' \
        'created_at_weekday_sun_mon_tue + user_friends_count_per_day + num_media + ' \
        'user_profile_use_background_image + created_at_weekday + ' \
        'user_listed_count + created_at_hour + user_friends_count + user_created_at_delta + user_statuses_count',
    'features_extended_some_multiple':
        'fake ~ user_verified + text_num_caps + text_num_digits + user_screen_name_has_caps + user_screen_name_has_digits + num_urls_is_nonzero + ' \
        'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
        'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
        'user_name_has_weird_chars + user_default_profile + ' \
        'created_at_weekday_sun_mon_tue + created_at_hour_08_to_17 + user_friends_count_per_day + num_media + ' \
        'created_at_hour_18_to_00 + text_num_swears + user_profile_use_background_image + created_at_weekday + ' \
        'user_listed_count + created_at_hour + user_friends_count + user_created_at_delta + user_statuses_count',
    'features_extended_some_multiple_without_text_num_swears':
        'fake ~ user_verified + text_num_caps + text_num_digits + user_screen_name_has_caps + user_screen_name_has_digits + num_urls_is_nonzero + ' \
        'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
        'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
        'user_name_has_weird_chars + user_default_profile + ' \
        'created_at_weekday_sun_mon_tue + created_at_hour_08_to_17 + user_friends_count_per_day + num_media + ' \
        'created_at_hour_18_to_00 + user_profile_use_background_image + created_at_weekday + ' \
        'user_listed_count + created_at_hour + user_friends_count + user_created_at_delta + user_statuses_count',
    'features_extended_some_multiple_without_biasing_features':
        'fake ~ user_verified + text_num_caps + text_num_digits + user_screen_name_has_caps + user_screen_name_has_digits + num_urls_is_nonzero + ' \
        'user_description_num_exclam + user_followers_count_per_day + user_listed_count_per_day + user_followers_count + ' \
        'user_statuses_count_per_day + user_description_num_caps + user_favourites_count_per_day + ' \
        'user_name_has_weird_chars + user_default_profile + ' \
        'created_at_weekday_sun_mon_tue + user_friends_count_per_day + num_media + ' \
        'user_profile_use_background_image + created_at_weekday + ' \
        'user_listed_count + created_at_hour + user_friends_count + user_created_at_delta + user_statuses_count',
    'features_extended_all':
        'fake ~ tweet_id + retweet_count + user_verified + user_friends_count + user_followers_count + ' \
        'user_favourites_count + geo_coordinates + num_hashtags + num_mentions + num_urls + ' \
        'num_media + num_media_is_nonzero + ' \
        'text_num_caps + text_num_digits + text_num_nonstandard + text_num_nonstandard_extended + ' \
        'text_num_exclam + text_num_caps_exclam + text_num_caps_digits + text_num_caps_digits_exclam + ' \
        'text_num_swears + ' \
        'num_urls_is_nonzero + num_hashtags_is_nonzero + num_mentions_is_more_than_2 + ' \
        'created_at_hour + created_at_hour_08_to_17 + created_at_hour_18_to_00 + created_at_weekday + created_at_weekday_sun_mon_tue + ' \
        'created_at_hour_of_week + ' \
        'user_description_num_caps + user_description_num_digits + user_description_num_nonstandard + ' \
        'user_description_num_nonstandard_extended + user_description_num_exclam + ' \
        'user_description_num_caps_with_num_nonstandard + user_description_num_non_a_to_z + ' \
        'user_description_num_non_a_to_z_non_digits + user_description_num_caps_exclam + ' \
        'user_default_profile_image + ' \
        'user_listed_count + ' \
        'user_profile_use_background_image + user_default_profile + ' \
        'user_screen_name_has_caps + user_screen_name_has_digits + user_screen_name_has_underscores + ' \
        'user_screen_name_has_caps_digits + user_screen_name_has_caps_underscores + user_screen_name_has_digits_underscores + ' \
        'user_screen_name_has_caps_digits_underscores + user_screen_name_num_caps + user_screen_name_num_digits + ' \
        'user_screen_name_num_underscores + user_screen_name_num_caps_digits + user_screen_name_num_caps_underscores + ' \
        'user_screen_name_num_digits_underscores + user_screen_name_num_caps_digits_underscores + ' \
        'user_screen_name_has_weird_chars + user_screen_name_num_weird_chars + ' \
        'user_name_has_caps + user_name_has_digits + user_name_has_underscores + user_name_has_caps_digits + ' \
        'user_name_has_caps_underscores + user_name_has_digits_underscores + user_name_has_caps_digits_underscores + ' \
        'user_name_num_caps + user_name_num_digits + user_name_num_underscores + user_name_num_caps_digits + ' \
        'user_name_num_caps_underscores + user_name_num_digits_underscores + user_name_num_caps_digits_underscores + ' \
        'user_name_has_weird_chars + user_name_num_weird_chars + user_name_has_nonprintable_chars + ' \
        'user_name_num_nonprintable_chars + ' \
        'user_statuses_count + user_created_at_delta + user_statuses_count_per_day + user_followers_count_per_day + ' \
        'user_listed_count_per_day + user_friends_count_per_day + user_favourites_count_per_day + retweet_count_per_day',
    'features_extended_all_reduced':
        'fake ~ user_verified + ' \
        'geo_coordinates + num_hashtags + num_mentions + num_urls + ' \
        'num_media + num_media_is_nonzero + ' \
        'text_num_caps + text_num_digits + text_num_nonstandard + text_num_nonstandard_extended + ' \
        'text_num_exclam + text_num_caps_exclam + text_num_caps_digits + text_num_caps_digits_exclam + ' \
        'text_num_swears + ' \
        'num_urls_is_nonzero + num_hashtags_is_nonzero + num_mentions_is_more_than_2 + ' \
        'created_at_hour + created_at_hour_08_to_17 + created_at_hour_18_to_00 + created_at_weekday + created_at_weekday_sun_mon_tue + ' \
        'created_at_hour_of_week + ' \
        'user_description_num_caps + user_description_num_digits + user_description_num_nonstandard + ' \
        'user_description_num_nonstandard_extended + user_description_num_exclam + ' \
        'user_description_num_caps_with_num_nonstandard + user_description_num_non_a_to_z + ' \
        'user_description_num_non_a_to_z_non_digits + user_description_num_caps_exclam + ' \
        'user_default_profile_image + ' \
        'user_profile_use_background_image + user_default_profile + ' \
        'user_screen_name_has_caps + user_screen_name_has_digits + user_screen_name_has_underscores + ' \
        'user_screen_name_has_caps_digits + user_screen_name_has_caps_underscores + user_screen_name_has_digits_underscores + ' \
        'user_screen_name_has_caps_digits_underscores + user_screen_name_num_caps + user_screen_name_num_digits + ' \
        'user_screen_name_num_underscores + user_screen_name_num_caps_digits + user_screen_name_num_caps_underscores + ' \
        'user_screen_name_num_digits_underscores + user_screen_name_num_caps_digits_underscores + ' \
        'user_screen_name_has_weird_chars + user_screen_name_num_weird_chars + ' \
        'user_name_has_caps + user_name_has_digits + user_name_has_underscores + user_name_has_caps_digits + ' \
        'user_name_has_caps_underscores + user_name_has_digits_underscores + user_name_has_caps_digits_underscores + ' \
        'user_name_num_caps + user_name_num_digits + user_name_num_underscores + user_name_num_caps_digits + ' \
        'user_name_num_caps_underscores + user_name_num_digits_underscores + user_name_num_caps_digits_underscores + ' \
        'user_name_has_weird_chars + user_name_num_weird_chars + user_name_has_nonprintable_chars + ' \
        'user_name_num_nonprintable_chars + ' \
        'user_created_at_delta + user_statuses_count_per_day + user_followers_count_per_day + ' \
        'user_listed_count_per_day + user_friends_count_per_day + user_favourites_count_per_day + retweet_count_per_day'
}

# define functions needed to compute features
def number_of_swears(text):
    number = 0
    swearwords = ['fuck', 'shit', 'dumb', 'retard', 'kill', 'crap']
    for swearword in swearwords:
        number += len(re.findall(swearword, text.lower()))
    return number
def get_weekday_number_from_text(text):
    weekdays = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}
    weekday_text = re.search('^([A-Z][a-z][a-z])\s', text).group(1)
    weekday_number = weekdays[weekday_text]
    utc_hour = int(re.search('\s([0-9][0-9]):', text).group(1))
    if utc_hour - 5 < 0:  # although in the UK the day number is 'weekday_number', in the US the day number might be one less due to timezone shift
        weekday_number -= 1
    if weekday_number == 0:
        weekday_number = 7
    return weekday_number
def get_time_delta(datetime_created_string):
    datetime_today = datetime.datetime.now()
    datetime_created = datetime.datetime.strptime(datetime_created_string, "%a %b %d %H:%M:%S +0000 %Y")
    datetime_delta = datetime_today - datetime_created
    return datetime_delta.days
def get_est_hour_from_text(text):
    utc_hour = int(re.search('\s([0-9][0-9]):', text).group(1))
    est_hour = (utc_hour + 24 - 5) % 24
    return est_hour
def get_hour_of_week_from_text(text):
    hour_of_week = (get_weekday_number_from_text(text) - 1) * 24 + get_est_hour_from_text(text)
    return hour_of_week

# add derived features related to various base features
def add_derived_features(data):

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
    data['user_screen_name_num_digits_underscores'] = data['user_screen_name'].apply(lambda text: len(re.findall('[0-9_]', text)))
    data['user_screen_name_num_caps_digits_underscores'] = data['user_screen_name'].apply(lambda text: len(re.findall('[A-Z0-9_]', text)))
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
    data['created_at_hour'] = data['created_at'].apply(get_est_hour_from_text)
    data['created_at_hour_18_to_00'] = data['created_at_hour'].isin([18, 19, 20, 21, 22, 23, 0]).astype(int)
    data['created_at_hour_08_to_17'] = data['created_at_hour'].isin(range(8, 17)).astype(int)
    data['created_at_weekday'] = data['created_at'].apply(get_weekday_number_from_text)
    data['created_at_weekday_sun_mon_tue'] = data['created_at_weekday'].isin([7, 1, 2]).astype(int)
    data['created_at_hour_of_week'] = data['created_at'].apply(get_hour_of_week_from_text)

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
    data['user_name_num_digits_underscores'] = data['user_name'].apply(lambda text: len(re.findall('[0-9_]', text)))
    data['user_name_num_caps_digits_underscores'] = data['user_name'].apply(lambda text: len(re.findall('[A-Z0-9_]', text)))
    data['user_name_has_weird_chars'] = data['user_name'].apply(lambda text: int(len(re.findall('[^A-Za-z .\']', text)) >= 1))
    data['user_name_num_weird_chars'] = data['user_name'].apply(lambda text: len(re.findall('[^A-Za-z .\']', text)))
    data['user_name_has_nonprintable_chars'] = data['user_name'].apply(lambda text: int(len(re.findall('[^ -~]', text)) >= 1))
    data['user_name_num_nonprintable_chars'] = data['user_name'].apply(lambda text: len(re.findall('[^ -~]', text)))

    # nonzero number features
    data['num_urls_is_nonzero'] = data['num_urls'].apply(lambda number: int(number >= 1))
    data['num_media_is_nonzero'] = data['num_media'].apply(lambda number: int(number >= 1))
    data['num_hashtags_is_nonzero'] = data['num_hashtags'].apply(lambda number: int(number >= 1))
    data['num_mentions_is_more_than_2'] = data['num_hashtags'].apply(lambda number: int(number > 2))

    # per-unit-time related features
    data['user_created_at_delta'] = data['user_created_at'].apply(get_time_delta)  # number of days from account creation to now
    data['created_at_delta'] = data['user_created_at'].apply(get_time_delta)  # number of days from tweet creation to now
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
    data['user_verified'] = data['user_verified'].astype(int)

    return data

# SVM functions
def compute_svm_performance(writing_queue, X, y, SVM_KERNEL, SVM_POLY_DEGREE, SVM_MAX_ITER, SVM_C):

    # initialise scores lists
    accuracy_scores, roc_auc_scores, confusion_matrices, classification_reports, precision_scores, recall_scores, f1_scores = [], [], [], [], [], [], []

    # get Numpy arrays from Pandas dataframes
    y_array, X_array = np.ravel(y.values), X.values

    # use a Stratified K-Fold cross-validation method with minority class upsampling during training
    cv = StratifiedKFold(n_splits=CV_NUMBER_OF_SPLITS, shuffle=True, random_state=123)

    for train_index, test_index in cv.split(X_array, y_array):
        # select train and test data based on indices from the Stratified K-Fold Cross-Validation function
        X_train, X_test, y_train, y_test = X.iloc[train_index, :], X.iloc[test_index, :], y.iloc[train_index, :], y.iloc[test_index, :]

        # up/down-sample data to balance the training data set
        X_train_array, X_test_array, y_train_array, y_test_array = resample_data(X_train, X_test, y_train, y_test)

        # fit model
        # model = svm.SVC(C=C, kernel='rbf', probability=True, verbose=True)
        model = svm.SVC(C=SVM_C, kernel=SVM_KERNEL, probability=True, verbose=False, max_iter=SVM_MAX_ITER, degree=SVM_POLY_DEGREE, cache_size=SVM_CACHE_SIZE)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            model.fit(X_train_array, y_train_array)

        # calculate performances for one fold
        y_pred = model.predict(X_test_array)
        y_pred_proba = model.predict_proba(X_test_array)
        accuracy_scores.append(metrics.accuracy_score(y_test_array, y_pred))
        roc_auc_scores.append(metrics.roc_auc_score(y_test_array, y_pred_proba[:, 1]))
        confusion_matrices.append(metrics.confusion_matrix(y_test_array, y_pred))
        classification_reports.append(metrics.classification_report(y_test_array, y_pred))
        precision_scores.append(metrics.precision_score(y_test_array, y_pred))
        recall_scores.append(metrics.recall_score(y_test_array, y_pred))
        f1_scores.append(metrics.f1_score(y_test_array, y_pred))

    # calculate averaged performances (from all folds)
    mean_accuracy_score = np.mean(accuracy_scores)
    mean_roc_auc_score = np.mean(roc_auc_scores)
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
    mean_precision_score = np.mean(precision_scores)
    mean_recall_score = np.mean(recall_scores)
    mean_f1_score = np.mean(f1_scores)

    results = [
        str(datetime.datetime.now()), MODEL_NAME, SVM_KERNEL, str(SVM_MAX_ITER),
        str(SVM_POLY_DEGREE), format(SVM_C, '.0E'), format(mean_accuracy_score, '0.10f'),
        format(mean_roc_auc_score, '0.10f'), format(mean_precision_score, '0.10f'), format(mean_recall_score, '0.10f'),
        format(mean_f1_score, '0.10f'), str(np.around(mean_confusion_matrix, 2)[0,0]),
        str(np.around(mean_confusion_matrix, 2)[0,1]), str(np.around(mean_confusion_matrix, 2)[1,0]),
        str(np.around(mean_confusion_matrix, 2)[1,1])
    ]

    writing_queue.put(results)
    return 'done'
def svm_writer(writing_queue):
    # open file to write model performances into
    performances_file = open(PERFORMANCES_FILE_NAME, 'wb')
    performances_writer = csv.writer(performances_file)

    # write header to performances file
    performances_writer.writerow(
        ["datetime", "model", "kernel", "max_iter", "poly_degree", "C", "mean_accuracy_score", "mean_roc_auc_score",
         "mean_precision_score", "mean_recall_score", "mean_f1_score", "mean_cm_TN", "mean_cm_FP",
         "mean_cm_FN", "mean_cm_TP"]
    )

    # keep writing into file data from queue
    while True:
        # wait for result to appear in the queue
        results = writing_queue.get()

        # if got signal 'kill' exit the loop
        if results == 'kill':
            break

        # print averaged performances (from all folds) to terminal
        print("\t".join(results))

        # print averaged performances (from all folds) into CSV file
        performances_writer.writerow(results)

    # close file into which the model performances have been written
    performances_file.close()

def compute_logistic_regression_performance(X, y):

    # initialise scores lists
    accuracy_scores, roc_auc_scores, confusion_matrices, classification_reports, precision_scores, recall_scores, f1_scores, weights_list = [], [], [], [], [], [], [], []

    # get Numpy arrays from Pandas dataframes
    y_array, X_array = np.ravel(y.values), X.values

    # use a Stratified K-Fold cross-validation method with minority class upsampling during training
    cv = StratifiedKFold(n_splits=CV_NUMBER_OF_SPLITS, shuffle=True, random_state=123)

    for train_index, test_index in cv.split(X_array, y_array):
        # select train and test data based on indices from the Stratified K-Fold Cross-Validation function
        X_train, X_test, y_train, y_test = X.iloc[train_index, :], X.iloc[test_index, :], y.iloc[train_index, :], y.iloc[test_index, :]

        # up/down-sample data to balance the training data set
        X_train_array, X_test_array, y_train_array, y_test_array = resample_data(X_train, X_test, y_train, y_test)

        # fit model
        # model = LogisticRegression(solver='liblinear', penalty='l1', verbose=1, tol=1.3)
        model = LogisticRegression(solver='liblinear', penalty='l1')
        model.fit(X_train_array, y_train_array)

        # calculate performances for one fold
        y_pred = model.predict(X_test_array)
        y_pred_proba = model.predict_proba(X_test_array)
        accuracy_scores.append(metrics.accuracy_score(y_test_array, y_pred))
        roc_auc_scores.append(metrics.roc_auc_score(y_test_array, y_pred_proba[:, 1]))
        confusion_matrices.append(metrics.confusion_matrix(y_test_array, y_pred))
        classification_reports.append(metrics.classification_report(y_test_array, y_pred))
        precision_scores.append(metrics.precision_score(y_test_array, y_pred))
        recall_scores.append(metrics.recall_score(y_test_array, y_pred))
        f1_scores.append(metrics.f1_score(y_test_array, y_pred))
        weights_list.append([coeff for coeff in (model.coef_)[0]])

    mean_accuracy_score = np.mean(accuracy_scores)
    mean_roc_auc_score = np.mean(roc_auc_scores)
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
    mean_precision_score = np.mean(precision_scores)
    mean_recall_score = np.mean(recall_scores)
    mean_f1_score = np.mean(f1_scores)
    weights_mean = np.mean(weights_list, axis=0)
    features_weights = [X.columns.tolist(), weights_mean.tolist()]
    features_weights = map(list, zip(*features_weights))
    features_weights_sorted = sorted(features_weights, key=lambda column: abs(column[1]), reverse=True)

    results = [
        format(mean_accuracy_score, '0.10f'), format(mean_roc_auc_score, '0.10f'),
        format(mean_precision_score, '0.10f'),
        format(mean_recall_score, '0.10f'), format(mean_f1_score, '0.10f'),
        str(np.around(mean_confusion_matrix, 2)[0, 0]),
        str(np.around(mean_confusion_matrix, 2)[0, 1]), str(np.around(mean_confusion_matrix, 2)[1, 0]),
        str(np.around(mean_confusion_matrix, 2)[1, 1]), str(features_weights_sorted)
    ]

    print("\t".join(results))

def resample_data(X_train, X_test, y_train, y_test):
    data_train = pd.concat([y_train, X_train], axis=1)

    if RESAMPLE == 'up':
        # in the training set, upsample minority class (ie fake news class) so that it has the same number of rows as the majority class (ie real news class)
        data_train_real = data_train[data_train.fake == 0]
        data_train_fake = data_train[data_train.fake == 1]
        data_train_fake_upsampled = resample(data_train_fake, replace=True, n_samples=data_train_real.shape[0], random_state=123)
        data_train_upsampled = pd.concat([data_train_real, data_train_fake_upsampled])
        y_train_upsampled, X_train_upsampled = dmatrices(features, data_train_upsampled, return_type='dataframe')

        # convert pandas dataframes into numpy arrays
        y_train_upsampled_array, X_train_upsampled_array = np.ravel(y_train_upsampled.values), X_train_upsampled.values
        y_test_array, X_test_array = np.ravel(y_test.values), X_test.values

        X_train_array, X_test_array, y_train_array, y_test_array = X_train_upsampled_array, X_test_array, y_train_upsampled_array, y_test_array

    if RESAMPLE == 'down':
        # in the training set, downsample majority class (ie other news class) so that it has the same number of rows as the minority class (ie fake news class)
        data_train_real = data_train[data_train.fake == 0]
        data_train_fake = data_train[data_train.fake == 1]
        data_train_real_downsampled = resample(data_train_real, replace=True, n_samples=data_train_fake.shape[0], random_state=123)
        data_train_downsampled = pd.concat([data_train_fake, data_train_real_downsampled])
        y_train_downsampled, X_train_downsampled = dmatrices(features, data_train_downsampled, return_type='dataframe')

        # convert pandas dataframes into numpy arrays
        y_train_downsampled_array, X_train_downsampled_array = np.ravel(y_train_downsampled.values), X_train_downsampled.values
        y_test_array, X_test_array = np.ravel(y_test.values), X_test.values

        X_train_array, X_test_array, y_train_array, y_test_array = X_train_downsampled_array, X_test_array, y_train_downsampled_array, y_test_array

    if RESAMPLE == 'none':
        y_train_array, X_train_array = np.ravel(y_train.values), X_train.values
        y_test_array, X_test_array = np.ravel(y_test.values), X_test.values

    return X_train_array, X_test_array, y_train_array, y_test_array

def compute_knn_performance(X, y, KNN_N_NEIGHBORS):

    # initialise scores lists
    accuracy_scores, roc_auc_scores, confusion_matrices, classification_reports, precision_scores, recall_scores, f1_scores = [], [], [], [], [], [], []

    # get Numpy arrays from Pandas dataframes
    y_array, X_array = np.ravel(y.values), X.values

    # use a Stratified K-Fold cross-validation method with minority class upsampling during training
    cv = StratifiedKFold(n_splits=CV_NUMBER_OF_SPLITS, shuffle=True, random_state=123)

    for train_index, test_index in cv.split(X_array, y_array):
        # select train and test data based on indices from the Stratified K-Fold Cross-Validation function
        X_train, X_test, y_train, y_test = X.iloc[train_index, :], X.iloc[test_index, :], y.iloc[train_index, :], y.iloc[test_index, :]

        # up/down-sample data to balance the training data set
        X_train_array, X_test_array, y_train_array, y_test_array = resample_data(X_train, X_test, y_train, y_test)

        # fit model
        model = KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS, n_jobs=-1)
        model.fit(X_train_array, y_train_array)

        # calculate performances for one fold
        y_pred = model.predict(X_test_array)
        y_pred_proba = model.predict_proba(X_test_array)
        accuracy_scores.append(metrics.accuracy_score(y_test_array, y_pred))
        roc_auc_scores.append(metrics.roc_auc_score(y_test_array, y_pred_proba[:, 1]))
        confusion_matrices.append(metrics.confusion_matrix(y_test_array, y_pred))
        classification_reports.append(metrics.classification_report(y_test_array, y_pred))
        precision_scores.append(metrics.precision_score(y_test_array, y_pred))
        recall_scores.append(metrics.recall_score(y_test_array, y_pred))
        f1_scores.append(metrics.f1_score(y_test_array, y_pred))

    mean_accuracy_score = np.mean(accuracy_scores)
    mean_roc_auc_score = np.mean(roc_auc_scores)
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
    mean_precision_score = np.mean(precision_scores)
    mean_recall_score = np.mean(recall_scores)
    mean_f1_score = np.mean(f1_scores)

    results = [
        str(KNN_N_NEIGHBORS),
        format(mean_accuracy_score, '0.10f'), format(mean_roc_auc_score, '0.10f'),
        format(mean_precision_score, '0.10f'),
        format(mean_recall_score, '0.10f'), format(mean_f1_score, '0.10f'),
        str(np.around(mean_confusion_matrix, 2)[0, 0]),
        str(np.around(mean_confusion_matrix, 2)[0, 1]), str(np.around(mean_confusion_matrix, 2)[1, 0]),
        str(np.around(mean_confusion_matrix, 2)[1, 1])
    ]

    return results

if __name__ == '__main__':

    # import data from TSV file
    data = pd.read_csv('dataset_new_combined_20170804.tsv', sep='\t')
    data = data.dropna(subset = ['is_fake_news_2'])
    data = data.drop(data[data.is_fake_news_2 == 'UNKNOWN'].index)
    data['fake'] = (data.is_fake_news_2 == 'TRUE').astype(int)

    # add derived features related to various base features
    data = add_derived_features(data)

    # scale feature data so that each feature has mean of 0 and standard deviation of 1
    data_scaled = data
    data_scaled[statistics_features] = StandardScaler().fit_transform(data_scaled[statistics_features])
    if USE_SCALED_DATA:
        data = data_scaled

    if MODEL_NAME == 'log_reg':

        results_header = [
            "mean_accuracy_score", "mean_roc_auc_score", "mean_precision_score", "mean_recall_score", "mean_f1_score",
            "mean_cm_TN", "mean_cm_FP", "mean_cm_FN", "mean_cm_TP", "features_weights_sorted"
        ]
        print("\t".join(results_header))

        for LOG_REG_FEATURE_SET_NAME in FEATURES:

            # choose feature set given by FEATURES_TO_USE
            features = FEATURES[LOG_REG_FEATURE_SET_NAME]
            print(LOG_REG_FEATURE_SET_NAME)
            # get feature and label data based on 'features' variable
            y, X = dmatrices(features, data, return_type='dataframe')

            compute_logistic_regression_performance(X, y)

    if MODEL_NAME == 'svm':

        # choose feature set given by FEATURES_TO_USE
        features = FEATURES[SVM_FEATURES_TO_USE]
        # get feature and label data based on 'features' variable
        y, X = dmatrices(features, data, return_type='dataframe')

        # set up parallel processing (init writing queue, init jobs list, init processes pool, start svm_writer listener)
        manager = mp.Manager()
        writing_queue = manager.Queue()
        jobs = []
        pool = mp.Pool(processes=SVM_NUMBER_OF_PROCESSES + 1) # additional 1 process is for svm_writer which shouldn't take up much CPU power
        pool.apply_async(svm_writer, (writing_queue,))

        # calculate performances
        for SVM_KERNEL in SVM_KERNEL_LIST:
            SVM_POLY_DEGREE_LIST_TO_USE = SVM_POLY_DEGREE_LIST if SVM_KERNEL == 'poly' else [0]
            for SVM_POLY_DEGREE in SVM_POLY_DEGREE_LIST_TO_USE:
                for SVM_MAX_ITER in SVM_MAX_ITER_LIST:
                    for SVM_C in SVM_C_LIST:
                        jobs.append(pool.apply_async(compute_svm_performance, (writing_queue, X, y, SVM_KERNEL, SVM_POLY_DEGREE, SVM_MAX_ITER, SVM_C)))

        # clean up parallel processing (close pool, wait for processes to finish, kill writing_queue, wait for queue to be killed)
        pool.close()
        for job in jobs:
            job.get()
        writing_queue.put('kill')
        pool.join()
        print("DONE")

    if MODEL_NAME == 'knn':

        # open file to write model performances into
        performances_file = open(PERFORMANCES_FILE_NAME, 'wb')
        performances_writer = csv.writer(performances_file)

        # write header to terminal and performances file
        results_header = [
            "n_neighbors", "mean_accuracy_score", "mean_roc_auc_score", "mean_precision_score", "mean_recall_score",
            "mean_f1_score", "mean_cm_TN", "mean_cm_FP", "mean_cm_FN", "mean_cm_TP"
        ]
        print("\t".join(results_header))
        performances_writer.writerow(results_header)

        # choose feature set given by FEATURES_TO_USE
        features = FEATURES[KNN_FEATURES_TO_USE]
        # get feature and label data based on 'features' variable
        y, X = dmatrices(features, data, return_type='dataframe')

        for KNN_N_NEIGHBORS in KNN_N_NEIGHBORS_LIST:
            # calculate performances
            results = compute_knn_performance(X, y, KNN_N_NEIGHBORS)
            # print averaged performances (from all folds) to terminal and CSV file
            print("\t".join(results))
            performances_writer.writerow(results)

        # close file into which the model performances have been written
        performances_file.close()
