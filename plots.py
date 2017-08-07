# (c) 2017 Marian Longa

#TODO: change histogram to bar chart
#TODO: choose the features for discrete and continuous plotting more wisely and more of them

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import re
import datetime

sns.set(color_codes=True)

INPUT_FILE_NAME      = 'dataset_new_combined_20170804.tsv'
PLOTS_PATH           = 'plots/'
STATISTICS_FILE_NAME = 'statistics.txt'
DO_DISCRETE_PLOTS    = False
DO_CONTINUOUS_PLOTS  = True
DO_STATISTICS        = False

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
        plt.xlabel(feature)
        plt.ylabel("normalised density of tweets")
        plt.savefig(PLOTS_PATH + feature + '.png')
        plt.close()
    print("Continuous features plotted successfully")

# plot discrete features
if DO_DISCRETE_PLOTS:
    discrete_features = ['user_verified', 'geo_coordinates', 'num_hashtags', 'num_mentions',
                         'num_urls', 'num_media',
                         'created_at_hour', 'created_at_hour_23_to_5', 'created_at_hour_13_to_22', 'created_at_weekday',
                         'created_at_weekday_sun_mon_tue', 'user_default_profile_image',
                         'user_name_has_digits_underscores', 'user_profile_use_background_image', 'user_default_profile',
                         'user_name_has_weird_chars', 'user_name_num_weird_chars', 'user_name_has_nonprintable_chars',
                         'user_name_num_nonprintable_chars', 'user_name_num_caps']
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
        'created_at_hour', 'created_at_hour_23_to_5', 'created_at_hour_13_to_22', 'created_at_weekday', 'created_at_weekday_sun_mon_tue',
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
    statistics = []
    for feature in statistics_features:
        t, p = stats.ttest_ind(data_fake[feature].values, data_other[feature].values)
        statistics.append([feature, format(float(t), '+.20f'), format(p, '.20f')])
    statistics_sorted = sorted(statistics, key=lambda s: s[2])

    # print statistics
    statistics_file = open(STATISTICS_FILE_NAME, 'w')
    template = '{0:60} {1:30} {2:30}'
    statistics_file.write(template.format("FEATURE", "T VALUE", "P VALUE"))
    statistics_file.write("\n")
    for statistic in statistics_sorted:
        statistics_file.write(template.format(*statistic) + "\n")
    statistics_file.close()
    print("Statistics written successfully")