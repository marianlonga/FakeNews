
# coding: utf-8

# In[4]:

from pymongo import MongoClient
import unicodecsv as csv

client = MongoClient("mongodb://python:DSI2017@146.169.33.32:27020/Twitter_TRUMPONEM")

db = client['Twitter_TRUMPONEM']
coll = db['tweets_onemonth']

## get all tweets in the DB
alltweets = coll.find()
print("ALL: " + str(coll.count())) ## total number of tweets in dataset


# In[5]:

## indicate how to obtain the dataset
pipeline = [
    {"$match": {"retweeted_status.retweet_count": {"$gt": 1000, "$lt": 100000}} },
    {"$group": { 
        "_id": "$retweeted_status.id", 
        "created_at" : {"$first": "$retweeted_status.created_at"},
        "retweet_count": { "$max": "$retweeted_status.retweet_count" }, # we take the one with the largest number of retweet_count
        "text": {"$first": "$retweeted_status.text" },
        "source": {"$first": "$retweeted_status.source" },
        "coordinates": {"$first": "$retweeted_status.coordinates" },
        "user_screen_name" : {"$first": "$retweeted_status.user.screen_name" },
        "user_verified" : {"$first": "$retweeted_status.user.verified"},
        "user_friends_count" : {"$first": "$retweeted_status.user.friends_count"},
        "user_followers_count" : {"$first": "$retweeted_status.user.followers_count"},
        "user_favourites_count" : {"$first": "$retweeted_status.user.favourites_count"},
        "num_hashtags" : {"$first": "$entities.hashtags"},
        "num_mentions" : {"$first": "$entities.user_mentions"},
        "num_urls" : {"$first": "$entities.urls"},
        "num_media" : {"$first": "$entities.media"}
        }},
    {"$project": {
        "created_at":1,
        "retweet_count":1,
        "text":1,
        "user_screen_name":1,
        "user_verified":1,
        "user_friends_count":1,
        "user_followers_count":1,
        "user_favourites_count":1,
        "source":1,
        "coordinates":{"$size": { "$ifNull": ["$num_hashtags", [] ] }},
        "num_hashtags" : {"$size": { "$ifNull": ["$num_hashtags", [] ] }},
        "num_mentions" : {"$size": { "$ifNull": ["$num_mentions", [] ] }},
        "num_urls" : {"$size": { "$ifNull": ["$num_urls", [] ] }},
        "num_media" : {"$size": { "$ifNull": ["$num_media", [] ] }}
        }},
    {"$sort": {"_id": 1}}
]


# In[6]:

## extract our dataset
viralTweets = coll.aggregate(pipeline)
## write it to a file
with open("dataset_2.csv", "wb") as f:
    w = csv.writer(f, delimiter='\t', quotechar='"', encoding='utf-8')
    w.writerow(['tweet_id','created_at','retweet_count','text','user_screen_name','user_verified','user_friends_count','user_followers_count','user_favourites_count', 'tweet_source', 'geo_coordinates','num_hashtags','num_mentions','num_urls','num_media'])
    for d in viralTweets:
        w.writerow([
                d['_id'],
                d['created_at'],
                d['retweet_count'],
                d['text'].replace('\t',' ').replace('\n',' ').replace('\"','\''),
                d['user_screen_name'],
                d['user_verified'],
                d['user_friends_count'],
                d['user_followers_count'],
                d['user_favourites_count'],
                d['source'],
                d['coordinates'],
                d['num_hashtags'],
                d['num_mentions'],
                d['num_urls'],
                d['num_media']
        ])


# In[ ]:



