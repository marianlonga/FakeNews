#TODO: change histogram to bar chart
#TODO: change current histograms to continuous versions for continuous features (eg num followers)
#TODO: add derived features (eg num caps in username)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

sns.set(color_codes=True)

INPUT_FILE_NAME = '/Users/longaster/Desktop/UROP2017/fakenews/marian/twitter_data_OneMonth_20170731.csv'

# read csv file
data = pd.read_csv(INPUT_FILE_NAME)
data = data.dropna(subset=['is_fake_news_2'])
data = data.drop((data['is_fake_news_2'] == 'UNKNOWN').astype(int))
data['fake'] = (data['is_fake_news_2'] == 'TRUE').astype(int)
data['user_verified'] = data['user_verified'].astype(int)
data['num_urls'] = data['num_urls'].astype(int)

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
    plt.savefig('plots/' + feature + '.png')

# plot discrete features
discrete_features = ['user_verified', 'geo_coordinates_available', 'num_hashtags', 'num_mentions',
                     'num_urls', 'num_media']
weights_fake = np.ones(data_fake.shape[0])/len(data_fake)
weights_other = np.ones(data_other.shape[0])/len(data_other)
weights = [weights_fake, weights_other]
colors = ['red', 'green']
for feature in discrete_features:
    plt.figure(figsize=(10, 5))
    plt.hist([np.ravel(data_fake[feature]), np.ravel(data_other[feature])], weights=weights, color=colors)
    plt.xlabel(feature)
    plt.ylabel("normalised density of tweets")
    plt.title("Distribution of tweets by feature '" + feature + "'")
    plt.savefig('plots/' + feature + '.png')

# calculate statistics
statistics_features = ['tweet_id', 'retweet_count', 'user_verified', 'user_friends_count', 'user_followers_count',
            'user_favourites_count', 'geo_coordinates_available', 'num_hashtags',
            'num_mentions', 'num_urls', 'num_media']
statistics = []
for feature in statistics_features:
    t, p = stats.ttest_ind(data_fake[feature].values, data_other[feature].values)
    statistics.append([feature, t, p])

# print statistics
template = '{0:30} {1:30} {2:30}'
print(template.format("FEATURE", "T TEST", "P VALUE"))
for statistic in statistics:
    print(template.format(*statistic))

