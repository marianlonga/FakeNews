# (c) 2017 Marian Longa

#TODO: change histogram to bar chart

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import re
import datetime
from pprint import pprint
import sys
import xlsxwriter
import random

sns.set(color_codes=True)

INPUT_FILE_NAME             = '/Users/longaster/Desktop/UROP2017/fakenews/marian/dataset_new_combined_20170804.tsv'
PLOTS_PATH                  = '/Users/longaster/Desktop/UROP2017/fakenews/marian/plots/'
STATISTICS_CSV_FILE_NAME    = '/Users/longaster/Desktop/UROP2017/fakenews/marian/statistics.csv'
STATISTICS_XLSX_FILE_NAME   = '/Users/longaster/Desktop/UROP2017/fakenews/marian/statistics.xlsx'
CONDENSED_STATISTICS_XLSX_FILE_NAME = '/Users/longaster/Desktop/UROP2017/fakenews/marian/statistics_condensed.xlsx'
DO_DISCRETE_PLOTS           = False
DO_CONTINUOUS_PLOTS         = False
DO_STATISTICS               = True
DIFF_MEAN_PERCENTAGE_CUTOFF = 0.05
P_VALUE_CUTOFF              = 0.01
INCLUDE_TEXT_FEATURES       = True

# read csv file
data = pd.read_csv(INPUT_FILE_NAME, sep='\t')
data = data.dropna(subset=['is_fake_news_2'])
data = data[data['is_fake_news_2'] != 'UNKNOWN']
data['fake'] = (data['is_fake_news_2'] == 'TRUE').astype(int)
data['user_verified'] = data['user_verified'].astype(int)
data['num_urls'] = data['num_urls'].astype(int)


# add derived features related to various base features

# derived feature functions
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
    if utc_hour - 5 < 0: # although in the UK the day number is 'weekday_number', in the US the day number might be one less due to timezone shift
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
data['user_name_num_digits_underscores'] = data['user_name'].apply(lambda text: len(re.findall('[0-9_]', text)) )
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


# divide data set into fake and other news
data_fake = data[data['fake'] == 1]
data_other = data[data['fake'] == 0]

# plot continuous features
if DO_CONTINUOUS_PLOTS:
    continuous_features = [
        'retweet_count', 'user_friends_count', 'user_followers_count',
        'user_favourites_count', 'user_listed_count', 'user_statuses_count', 'user_created_at_delta',
        'user_statuses_count_per_day',
        'user_created_at_delta', 'created_at_delta', 'user_statuses_count_per_day', 'user_followers_count_per_day',
        'user_listed_count_per_day', 'user_friends_count_per_day', 'user_favourites_count_per_day',
        'retweet_count_per_day']
    for feature in continuous_features:
        fig, ax = plt.subplots()
        sns.kdeplot(np.log10(data_fake[feature][data_fake[feature] != 0]), shade=True, ax=ax, color='r', legend=False)
        sns.kdeplot(np.log10(data_other[feature][data_other[feature] != 0]), shade=True, ax=ax, color='g', legend=False)
        plt.title("Distribution of tweets by feature '" + feature + "'")
        plt.xlabel("log10(" + feature + ")")
        plt.ylabel("normalised density of tweets")
        plt.savefig(PLOTS_PATH + feature + '.png')
        plt.close()
    print("Continuous features plotted successfully")

# plot discrete features
if DO_DISCRETE_PLOTS:
    discrete_features = [
        'user_verified', 'geo_coordinates', 'num_hashtags', 'num_mentions',
        'num_urls', 'num_media',
        'created_at_hour', 'created_at_hour_08_to_17', 'created_at_hour_18_to_00', 'created_at_weekday',
        'created_at_weekday_sun_mon_tue', 'created_at_hour_of_week', 'user_default_profile_image',
        'user_name_has_digits_underscores', 'user_profile_use_background_image', 'user_default_profile',
        'user_name_has_weird_chars', 'user_name_num_weird_chars', 'user_name_has_nonprintable_chars',
        'user_name_num_nonprintable_chars', 'user_name_num_caps'
    ]
    weights_fake = np.ones(data_fake.shape[0])/len(data_fake)
    weights_other = np.ones(data_other.shape[0])/len(data_other)
    weights = [weights_fake, weights_other]
    colors = ['red', 'green']
    for feature in discrete_features:
        plt.figure(figsize=(10, 5))
        plt.hist([np.ravel(data_fake[feature]), np.ravel(data_other[feature])], weights=weights, color=colors, bins=len(np.unique(data[feature])))
        plt.xlabel(feature)
        plt.ylabel("normalised density of tweets")
        plt.title("Distribution of tweets by feature '" + feature + "'")
        plt.savefig(PLOTS_PATH + feature + '.png')
        plt.close()
    print("Discrete features plotted successfully")

# calculate statistics
if DO_STATISTICS:
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
    feature_groups = [
        ['tweet_id'],
        ['retweet_count', 'retweet_count_per_day'],
        ['user_verified'],
        ['user_friends_count', 'user_friends_count_per_day'],
        ['user_followers_count', 'user_followers_count_per_day'],
        ['user_favourites_count', 'user_favourites_count_per_day'],
        ['geo_coordinates'],
        ['num_hashtags', 'num_hashtags_is_nonzero'],
        ['num_mentions', 'num_mentions_is_more_than_2'],
        ['num_urls', 'num_urls_is_nonzero'],
        ['num_media', 'num_media_is_nonzero'],
        ['text_num_caps', 'text_num_digits', 'text_num_nonstandard', 'text_num_nonstandard_extended',
        'text_num_exclam', 'text_num_caps_exclam', 'text_num_caps_digits', 'text_num_caps_digits_exclam',
        'text_num_swears'],
        ['created_at_hour', 'created_at_hour_08_to_17', 'created_at_hour_18_to_00'],
        ['created_at_weekday', 'created_at_weekday_sun_mon_tue'],
        ['created_at_hour_of_week'],
        ['user_description_num_caps', 'user_description_num_digits', 'user_description_num_nonstandard',
        'user_description_num_nonstandard_extended', 'user_description_num_exclam',
        'user_description_num_caps_with_num_nonstandard', 'user_description_num_non_a_to_z',
        'user_description_num_non_a_to_z_non_digits', 'user_description_num_caps_exclam'],
        ['user_default_profile_image'],
        ['user_listed_count', 'user_listed_count_per_day'],
        ['user_profile_use_background_image'],
        ['user_default_profile'],
        ['user_screen_name_has_caps', 'user_screen_name_has_digits', 'user_screen_name_has_underscores',
        'user_screen_name_has_caps_digits', 'user_screen_name_has_caps_underscores', 'user_screen_name_has_digits_underscores',
        'user_screen_name_has_caps_digits_underscores', 'user_screen_name_num_caps', 'user_screen_name_num_digits',
        'user_screen_name_num_underscores', 'user_screen_name_num_caps_digits', 'user_screen_name_num_caps_underscores',
        'user_screen_name_num_digits_underscores', 'user_screen_name_num_caps_digits_underscores',
        'user_screen_name_has_weird_chars', 'user_screen_name_num_weird_chars'],
        ['user_name_has_caps', 'user_name_has_digits', 'user_name_has_underscores', 'user_name_has_caps_digits',
        'user_name_has_caps_underscores', 'user_name_has_digits_underscores', 'user_name_has_caps_digits_underscores',
        'user_name_num_caps', 'user_name_num_digits', 'user_name_num_underscores', 'user_name_num_caps_digits',
        'user_name_num_caps_underscores', 'user_name_num_digits_underscores', 'user_name_num_caps_digits_underscores',
        'user_name_has_weird_chars', 'user_name_num_weird_chars', 'user_name_has_nonprintable_chars',
        'user_name_num_nonprintable_chars'],
        ['user_statuses_count', 'user_statuses_count_per_day'],
        ['user_created_at_delta']
    ]

    # calculate statistics and append them to a list
    statistics_current = []
    for feature in statistics_features:
        t_value, p_value = stats.ttest_ind(data_fake[feature].values, data_other[feature].values)
        diff_mean_value = data_fake[feature].mean() - data_other[feature].mean()
        diff_mean_percentage = diff_mean_value / data[feature].mean()
        statistics_current.append([feature, diff_mean_value, diff_mean_percentage, p_value, float(t_value)])
    if not INCLUDE_TEXT_FEATURES:
        text_features_prefixes = ['user_name', 'user_screen_name', 'user_description', 'text']
        statistics_current = [statistic for statistic in statistics_current if not statistic[0].startswith(tuple(text_features_prefixes))]
    statistics_all = statistics_current

    for statistic in statistics_all:
        statistic.append({'font_color': 'black'})

    # select which features to keep based on diff_mean_percentage and p_value
    statistics_current = [statistic for statistic in statistics_current if abs(statistic[2]) >= DIFF_MEAN_PERCENTAGE_CUTOFF]  # only keep statistics with big enough diff_mean_percentage value
    statistics_current = [statistic for statistic in statistics_current if statistic[3] <= P_VALUE_CUTOFF]  # only keep statistics with low p_value

    for statistic in statistics_all:
        if statistic not in statistics_current:
            statistic[5].update({'font_color': 'red'})

    # group features and select the best feature from each group
    statistics_current = sorted(statistics_current, key=lambda s: abs(s[2]), reverse=True)  # sort statistics by diff_mean_percentage
    statistics_with_best_feature_per_group = []
    for feature_group in feature_groups:
        statistics_group = [statistic for statistic in statistics_current if (statistic[0] in feature_group)]
        if statistics_group:
            statistics_with_best_feature_per_group.append(statistics_group[0])
    statistics_current = statistics_with_best_feature_per_group
    statistics_current = sorted(statistics_current, key=lambda s: abs(s[2]), reverse=True)  # sort statistics by diff_mean_percentage

    for statistic in statistics_all:
        if statistic in statistics_current:
            statistic[5].update({'bold': True})

    # annotate groups by different background colours
    for feature_group in feature_groups:
        color = '#' + ('%02x' % (random.randrange(106) + 150)) + ('%02x' % (random.randrange(106) + 150)) + ('%02x' % (random.randrange(106) + 150))
        for feature in feature_group:
            for statistic in statistics_all:
                #if statistic[0] == feature and statistic[5]['font_color] != 'red':
                if statistic[0] == feature:
                    statistic[5].update({'bg_color': color})

    # print statistics into CSV file
    statistics_csv_file = open(STATISTICS_CSV_FILE_NAME, 'w')
    statistics_csv_file.write("FEATURE,DIFF MEAN VALUE,DIFF MEAN PERCENTAGE,P VALUE,T VALUE\n")
    for statistic in statistics_current:
        statistic_formatted = statistic[0] + "," + format(statistic[1], '+50.20f') + "," + format(statistic[2], '+.20f') + "," + format(statistic[3], '.20f') + "," + format(statistic[4], '+.20f')
        statistics_csv_file.write(statistic_formatted + "\n")
    statistics_csv_file.close()

    # write data into Excel file
    statistics_all = sorted(statistics_all, key=lambda s: abs(s[2]), reverse=True)  # sort statistics by diff_mean_percentage
    for filename in [STATISTICS_XLSX_FILE_NAME, CONDENSED_STATISTICS_XLSX_FILE_NAME]:
        workbook = xlsxwriter.Workbook(filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column(0, 0, 40)
        worksheet.set_column(1, 1, 25)
        worksheet.set_column(2, 2, 20)
        worksheet.set_column(3, 3, 25)
        worksheet.set_column(4, 4, 20)

        format_bold = workbook.add_format({'bold': True})
        worksheet.write(0, 0, "FEATURE", format_bold)
        worksheet.write(0, 1, "DIFF MEAN VALUE", format_bold)
        worksheet.write(0, 2, "DIFF MEAN PERCENTAGE", format_bold)
        worksheet.write(0, 3, "P VALUE", format_bold)
        worksheet.write(0, 4, "T VALUE", format_bold)

        row = 1
        for statistic in statistics_all:
            if (filename == STATISTICS_XLSX_FILE_NAME) or (filename == CONDENSED_STATISTICS_XLSX_FILE_NAME and statistic[5]['font_color'] != 'red'):
                format0 = workbook.add_format(statistic[5])
                format1 = workbook.add_format(statistic[5])
                format1.set_num_format('0.0000000000')
                format2 = workbook.add_format(statistic[5])
                format2.set_num_format('0.0000000000')
                format3 = workbook.add_format(statistic[5])
                format3.set_num_format('0.00000000000000000000')
                format4 = workbook.add_format(statistic[5])
                format4.set_num_format('0.0000000000')
                worksheet.write(row, 0, statistic[0], format0)
                worksheet.write(row, 1, statistic[1], format1)
                worksheet.write(row, 2, statistic[2], format2)
                worksheet.write(row, 3, statistic[3], format3)
                worksheet.write(row, 4, statistic[4], format4)
                row += 1

        workbook.close()
