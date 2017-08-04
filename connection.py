
# coding: utf-8

from pymongo import MongoClient
import unicodecsv as csv

client = MongoClient("mongodb://python:DSI2017@146.169.33.32:27020/Twitter_TRUMPONEM")

db = client['Twitter_TRUMPONEM']
coll = db['tweets_onemonth']

## get all tweets in the DB
alltweets = coll.find()
print("ALL: " + str(coll.count())) ## total number of tweets in dataset

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
        "num_media" : {"$first": "$entities.media"},

        "user_default_profile_image" : {"$first": "$retweeted_status.user.default_profile_image"},
        "user_description" : {"$first": "$retweeted_status.user.description"},
        "user_listed_count" : {"$first": "$retweeted_status.user.listed_count"},
        "user_name" : {"$first": "$retweeted_status.user.name"},
        "user_profile_use_background_image" : {"$first": "$retweeted_status.user.profile_use_background_image"},
        "user_default_profile" : {"$first": "$retweeted_status.user.default_profile"},
        "user_statuses_count" : {"$first": "$retweeted_status.user.statuses_count"},
        "user_created_at" : {"$first": "$retweeted_status.user.created_at"}
        }},
    {"$project": {
        "created_at":1,
        "retweet_count":1,
        "text": {"$ifNull": ["$text", " "]},
        "user_screen_name": {"$ifNull": ["$user_screen_name", " "]},
        "user_verified":1,
        "user_friends_count":1,
        "user_followers_count":1,
        "user_favourites_count":1,
        "source": {"$ifNull": ["$source", " "]},
        "coordinates":{"$size": { "$ifNull": ["$num_hashtags", [] ] }},
        "num_hashtags" : {"$size": { "$ifNull": ["$num_hashtags", [] ] }},
        "num_mentions" : {"$size": { "$ifNull": ["$num_mentions", [] ] }},
        "num_urls" : {"$size": { "$ifNull": ["$num_urls", [] ] }},
        "num_media" : {"$size": { "$ifNull": ["$num_media", [] ] }},

        "user_default_profile_image" : 1,
        "user_description" : {"$ifNull": ["$user_description", " "]},
        "user_listed_count" : 1,
        "user_name" : {"$ifNull": ["$user_name", " "]},
        "user_profile_use_background_image" : 1,
        "user_default_profile" : 1,
        "user_statuses_count" : 1,
        "user_created_at" : 1
        }},
    {"$sort": {"_id": 1}}
]

## extract our dataset
viralTweets = coll.aggregate(pipeline)

## write it to a file
with open("dataset_new.tsv", "wb") as f:
    w = csv.writer(f, delimiter='\t', quotechar='"', encoding='utf-8')
    w.writerow(['tweet_id','created_at','retweet_count','text','user_screen_name','user_verified','user_friends_count',
                'user_followers_count','user_favourites_count', 'tweet_source', 'geo_coordinates','num_hashtags',
                'num_mentions','num_urls','num_media',     'user_default_profile_image','user_description',
                'user_listed_count','user_name', 'user_profile_use_background_image', 'user_default_profile',
                'user_statuses_count', 'user_created_at'])
    for d in viralTweets:
        w.writerow([
                d['_id'],
                d['created_at'],
                d['retweet_count'],
                d['text'].replace('\t',' ').replace('\n',' ').replace('\"','\''),
                d['user_screen_name'].replace('\t',' ').replace('\n',' ').replace('\"','\''),
                d['user_verified'],
                d['user_friends_count'],
                d['user_followers_count'],
                d['user_favourites_count'],
                d['source'].replace('\t',' ').replace('\n',' ').replace('\"','\''),
                d['coordinates'],
                d['num_hashtags'],
                d['num_mentions'],
                d['num_urls'],
                d['num_media'],

                d['user_default_profile_image'],
                d['user_description'].replace('\t',' ').replace('\n',' ').replace('\"','\''),
                d['user_listed_count'],
                d['user_name'].replace('\t',' ').replace('\n',' ').replace('\"','\''),
                d['user_profile_use_background_image'],
                d['user_default_profile'],
                d['user_statuses_count'],
                d['user_created_at']
        ])


print("Done")



