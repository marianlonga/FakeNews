# (c) 2017 Marian Longa
# v3: change positive correlation from real news to fake news

import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import resample
import sys
#from sklearn import linear_model
#from sklearn import train_test_split


# import data from TSV file
data = pd.read_csv('twitter_data_OneMonth_6000.tsv', sep='\t')
data = data.dropna(subset = ['is_fake_news_2'])
data = data.drop(data[data.is_fake_news_2 == 'UNKNOWN'].index)
data['fake'] = (data.is_fake_news_2 == 'TRUE').astype(int) # <-- NB changed FALSE to TRUE in this version ==> positive correlation corresponds to fake news not real news!
data['user_verified'] = data['user_verified'].astype(int)

print("**********")
print(data.shape)
print("**********")

#print(data.groupby('real').mean())
#print(data.groupby('real').var())

#scatter_plot = data[data.fake == 0].plot.scatter(x='num_hashtags', y='user_followers_count', color='DarkBlue')
#data[data.fake == 1].plot.scatter(x='num_hashtags', y='user_followers_count', color='DarkGreen', ax=scatter_plot)
#plt.show()


# upsample minority class (ie fake news class) so that it has the same number of rows as the majority class (ie real news class)
data_real = data[data.fake == 0] #print "data_real shape: ", data_real.shape[0]
data_fake = data[data.fake == 1] #print "data_fake shape: ", data_fake.shape[0]
data_fake_upsampled = resample(data_fake, replace=True, n_samples=data_real.shape[0], random_state=123)
data_upsampled = pd.concat([data_real, data_fake_upsampled])
print(data_upsampled.fake.value_counts())

print("@@@@@@@@")

#sys.exit()

y_original, X_original = dmatrices('fake ~ retweet_count + user_verified + user_friends_count + user_followers_count + user_favourites_count + num_hashtags + num_mentions + num_urls + num_media', data, return_type='dataframe')
y_upsampled, X_upsampled = dmatrices('fake ~ retweet_count + user_verified + user_friends_count + user_followers_count + user_favourites_count + num_hashtags + num_mentions + num_urls + num_media', data_upsampled, return_type='dataframe')
#y, X = dmatrices('real ~ user_verified + num_hashtags + num_mentions + num_urls + num_media', data, return_type='dataframe')
y_original = np.ravel(y_original)
y_upsampled = np.ravel(y_upsampled)

#X_train, X_test, y_train, y_test = X, X, y, y
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train, y_train = X_upsampled, y_upsampled
X_test,  y_test  = X_original,  y_original
#X_test, y_test = X_upsampled, y_upsampled




model = LogisticRegression(C=1e10, solver='newton-cg', max_iter=10000) #newton-cg is the only solver which seems to work with this data set
model.fit(X_train, y_train)


#print("model score: %f" % model.score(X, y))
print("model score (binary): %f" % metrics.accuracy_score(y_test, model.predict(X_test)))
print("model score (probabilistic): %f" % metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
print("mean fake news percentage in test set: %f" % y_test.mean())
print(metrics.confusion_matrix(y_test, model.predict(X_test)))
print(metrics.classification_report(y_test, model.predict(X_test)))


formatted_coeffs = ['{:0.15f}'.format(coeff) for coeff in (model.coef_)[0]]
print(pd.DataFrame(zip(X_train.columns, np.transpose(formatted_coeffs))))

print("")
print("")
pd.options.display.max_rows = 10
print(pd.DataFrame(model.predict_proba(X_train)))