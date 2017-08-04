import pandas as pd
import matplotlib.pyplot as plt
import sys


# import data from TSV file
data = pd.read_csv('twitter_data_OneMonth_6000.tsv', sep='\t')
data = data.dropna(subset = ['is_fake_news_2'])
data = data.drop(data[data.is_fake_news_2 == 'UNKNOWN'].index)
data['fake'] = (data.is_fake_news_2 == 'TRUE').astype(int) # <-- NB changed FALSE to TRUE in this version ==> positive correlation corresponds to fake news not real news!
data['user_verified'] = data['user_verified'].astype(int)
print(data.head())

sys.exit()

# plot
df_coeffs = pd.DataFrame({'user_verified': [0, 1], 'fake_real_density': model.coef_[0]})
df_coeffs = df_coeffs.set_index('coefficient')
coeffs_plot = df_coeffs.plot(kind='bar', title="Correlation between fake news and metadata", figsize=(8,6))
coeffs_plot.set_xlabel("metadata")
coeffs_plot.set_ylabel("weight")
plt.subplots_adjust(bottom=0.4)
plt.show()

# plot density of real/fake news vs user_verified
#df_user_verified = pd.DataFrame({'user_verified': [0, 1], ''})
#user_verified_plot = 