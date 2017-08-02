#TODO: change histogram to bar chart
#TODO: add more derived features

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import re

sns.set(color_codes=True)

INPUT_FILE_NAME = 'twitter_data_OneMonth_20170731.csv'
PLOTS_PATH = 'plots/'

# read csv file
data = pd.read_csv(INPUT_FILE_NAME)
data = data.dropna(subset=['is_fake_news_2'])
data = data[data['is_fake_news_2'] != 'UNKNOWN']
data['fake'] = (data['is_fake_news_2'] == 'TRUE').astype(int)
data['user_verified'] = data['user_verified'].astype(int)
data['num_urls'] = data['num_urls'].astype(int)


# add derived features
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

data['user_screen_name_number_of_digits']                           = data['user_screen_name'].apply(lambda text:     len(re.findall('[0-9]',     text)))
data['user_screen_name_has_digits']                                 = data['user_screen_name'].apply(lambda text: int(len(re.findall('[0-9]',     text)) >= 1))
data['user_screen_name_number_of_caps']                             = data['user_screen_name'].apply(lambda text:     len(re.findall('[A-Z]',     text)))
data['user_screen_name_has_caps']                                   = data['user_screen_name'].apply(lambda text: int(len(re.findall('[A-Z]',     text)) >= 1))
data['user_screen_name_number_of_special_chars']                    = data['user_screen_name'].apply(lambda text:     len(re.findall('[A-Z0-9]',  text)))
data['user_screen_name_has_special_chars']                          = data['user_screen_name'].apply(lambda text: int(len(re.findall('[A-Z0-9]',  text)) >= 1))
data['user_screen_name_number_of_special_chars_with_underscore']    = data['user_screen_name'].apply(lambda text:     len(re.findall('[A-Z0-9_]', text)))
data['user_screen_name_has_special_chars_with_underscore']          = data['user_screen_name'].apply(lambda text: int(len(re.findall('[A-Z0-9_]', text)) >= 1))
data['text_number_of_caps']                                         = data['text'].apply(lambda text:                 len(re.findall('[A-Z]',     text)))
data['text_has_caps']                                               = data['text'].apply(lambda text:             int(len(re.findall('[A-Z]',     text)) >= 1))
data['text_number_of_exclam']                                       = data['text'].apply(lambda text:                 len(re.findall('!',         text)))
data['text_has_exclam']                                             = data['text'].apply(lambda text:             int(len(re.findall('!',         text)) >= 1))
data['text_number_of_swears']                                       = data['text'].apply(number_of_swears)
data['created_at_hour']                                             = data['created_at'].apply(lambda text: int(re.search('\s([0-9][0-9]):', text).group(1)))
data['created_at_hour_23_to_5']                                     = data['created_at_hour'].isin([23, 0, 1, 2, 3, 4, 5])
data['created_at_hour_13_to_22']                                    = data['created_at_hour'].isin(range(13, 23))
data['created_at_weekday']                                          = data['created_at'].apply(extract_weekday_number)
data['created_at_weekday_sun_mon_tue']                              = data['created_at_weekday'].isin([7, 1, 2])

# divide data set into fake and other news
data_fake = data[data['fake'] == 1]
data_other = data[data['fake'] == 0]

# plot continuous features
continuous_features = ['retweet_count', 'user_friends_count', 'user_followers_count',
                       'user_favourites_count']
for feature in continuous_features:
    fig, ax = plt.subplots()
    sns.kdeplot(np.log10(data_fake[feature][data_fake[feature] != 0]), shade=True, ax=ax)
    sns.kdeplot(np.log10(data_other[feature][data_other[feature] != 0]), shade=True, ax=ax)
    plt.savefig(PLOTS_PATH + feature + '.png')

# plot discrete features
discrete_features = ['user_verified', 'geo_coordinates_available', 'num_hashtags', 'num_mentions',
                     'num_urls', 'num_media', 'user_screen_name_has_special_chars', 'text_number_of_exclam',
                     'created_at_hour', 'created_at_hour_23_to_5', 'created_at_hour_13_to_22', 'created_at_weekday',
                     'created_at_weekday_sun_mon_tue']
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

# calculate statistics
statistics_features = ['tweet_id', 'retweet_count', 'user_verified', 'user_friends_count', 'user_followers_count',
            'user_favourites_count', 'geo_coordinates_available', 'num_hashtags', 'num_mentions', 'num_urls',
            'num_media', 'user_screen_name_number_of_digits', 'user_screen_name_has_digits',
            'user_screen_name_number_of_caps', 'user_screen_name_has_caps', 'user_screen_name_number_of_special_chars',
            'user_screen_name_has_special_chars', 'user_screen_name_number_of_special_chars_with_underscore',
            'user_screen_name_has_special_chars_with_underscore', 'text_number_of_caps', 'text_has_caps',
            'text_number_of_exclam', 'text_has_exclam', 'text_number_of_swears', 'created_at_hour',
            'created_at_hour_23_to_5', 'created_at_hour_13_to_22', 'created_at_weekday', 'created_at_weekday_sun_mon_tue']
statistics = []
for feature in statistics_features:
    t, p = stats.ttest_ind(data_fake[feature].values, data_other[feature].values)
    statistics.append([feature, format(float(t), '+.20f'), format(p, '.20f')])
statistics_sorted = sorted(statistics, key=lambda s: s[2])

# print statistics
template = '{0:60} {1:30} {2:30}'
print(template.format("FEATURE", "T VALUE", "P VALUE"))
for statistic in statistics_sorted:
    print(template.format(*statistic))

