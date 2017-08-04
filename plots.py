#TODO: change histogram to bar chart
#TODO: add more derived features

#TODO: add derived features for 'user_description', 'user_name', etc features
#TODO: add feature for extracted year from 'user_created_at'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import re
import datetime

sns.set(color_codes=True)

#INPUT_FILE_NAME = 'twitter_data_OneMonth_20170731.csv'
INPUT_FILE_NAME = 'dataset_new_combined_20170804.tsv'
PLOTS_PATH = 'plots/'
STATISTICS_FILE_NAME = 'statistics.txt'

# read csv file
data = pd.read_csv(INPUT_FILE_NAME, sep='\t')
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
def get_user_created_at_delta(datetime_created_string):
    datetime_today = datetime.datetime.now()
    datetime_created = datetime.datetime.strptime(datetime_created_string, "%a %b %d %H:%M:%S +0000 %Y")
    datetime_delta = datetime_today - datetime_created
    return datetime_delta.days
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
data['user_description_has_special_chars']                          = data['user_description'].apply(lambda text: int(len(re.findall('[A-Z0-9]',  text)) >= 1))
data['user_name_has_digits_underscore']                             = data['user_name'].apply(lambda text: int(len(re.findall('[0-9]_',     text)) >= 1))
data['user_created_at_delta']                                       = data['user_created_at'].apply(get_user_created_at_delta)
data['user_statuses_count_per_day']                                 = data['user_statuses_count'] / data['user_created_at_delta']
data['user_name_has_weird_chars']                                   = data['user_name'].apply(lambda text: int(len(re.findall('[^A-Za-z .\']',     text)) >= 1)) # matches chars which should not be in a name
data['user_name_number_weird_chars']                                = data['user_name'].apply(lambda text: len(re.findall('[^A-Za-z .\']',     text))) # matches chars which should not be in a name
data['user_name_has_nonprintable_chars']                            = data['user_name'].apply(lambda text: int(len(re.findall('[^ -~]',     text)) >= 1)) # matches chars which should not be in a name
data['user_name_number_nonprintable_chars']                         = data['user_name'].apply(lambda text: len(re.findall('[^ -~]',     text)) >= 1) # matches chars which should not be in a name
data['user_description_has_weird_chars']                            = data['user_description'].apply(lambda text: int(len(re.findall('[^A-Za-z0-9., @#&]',  text)) >= 1))
data['user_description_number_weird_chars']                         = data['user_description'].apply(lambda text: len(re.findall('[^A-Za-z0-9., @#&]',  text)))
data['user_description_number_of_caps']                             = data['user_description'].apply(lambda text:     len(re.findall('[A-Z]',     text)))
data['user_name_number_of_caps']                                    = data['user_name'].apply(lambda text:     len(re.findall('[A-Z]',     text)))


# divide data set into fake and other news
data_fake = data[data['fake'] == 1]
data_other = data[data['fake'] == 0]

# plot continuous features
continuous_features = ['retweet_count', 'user_friends_count', 'user_followers_count',
                       'user_favourites_count', 'user_listed_count', 'user_statuses_count', 'user_created_at_delta',
                       'user_statuses_count_per_day']
for feature in continuous_features:
    fig, ax = plt.subplots()
    sns.kdeplot(np.log10(data_fake[feature][data_fake[feature] != 0]), shade=True, ax=ax)
    sns.kdeplot(np.log10(data_other[feature][data_other[feature] != 0]), shade=True, ax=ax)
    plt.savefig(PLOTS_PATH + feature + '.png')
    plt.close()
print("Continuous features plotted successfully")

# plot discrete features
discrete_features = ['user_verified', 'geo_coordinates', 'num_hashtags', 'num_mentions',
                     'num_urls', 'num_media', 'user_screen_name_has_special_chars', 'text_number_of_exclam',
                     'created_at_hour', 'created_at_hour_23_to_5', 'created_at_hour_13_to_22', 'created_at_weekday',
                     'created_at_weekday_sun_mon_tue', 'user_default_profile_image', 'user_description_has_special_chars',
                     'user_name_has_digits_underscore', 'user_profile_use_background_image', 'user_default_profile',
                     'user_name_has_weird_chars', 'user_name_number_weird_chars', 'user_name_has_nonprintable_chars',
                     'user_name_number_nonprintable_chars', 'user_description_has_weird_chars',
                     'user_description_number_weird_chars', 'user_description_number_of_caps',
                     'user_name_number_of_caps']
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
statistics_features = ['tweet_id', 'retweet_count', 'user_verified', 'user_friends_count', 'user_followers_count',
            'user_favourites_count', 'geo_coordinates', 'num_hashtags', 'num_mentions', 'num_urls',
            'num_media', 'user_screen_name_number_of_digits', 'user_screen_name_has_digits',
            'user_screen_name_number_of_caps', 'user_screen_name_has_caps', 'user_screen_name_number_of_special_chars',
            'user_screen_name_has_special_chars', 'user_screen_name_number_of_special_chars_with_underscore',
            'user_screen_name_has_special_chars_with_underscore', 'text_number_of_caps', 'text_has_caps',
            'text_number_of_exclam', 'text_has_exclam', 'text_number_of_swears', 'created_at_hour',
            'created_at_hour_23_to_5', 'created_at_hour_13_to_22', 'created_at_weekday', 'created_at_weekday_sun_mon_tue',
            'user_default_profile_image', 'user_description_has_special_chars', 'user_listed_count',
            'user_name_has_digits_underscore', 'user_profile_use_background_image', 'user_default_profile',
            'user_statuses_count', 'user_created_at_delta', 'user_statuses_count_per_day', 'user_name_has_weird_chars',
            'user_name_number_weird_chars', 'user_name_has_nonprintable_chars', 'user_name_number_nonprintable_chars',
            'user_description_has_weird_chars', 'user_description_number_weird_chars', 'user_description_number_of_caps',
            'user_name_number_of_caps']
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