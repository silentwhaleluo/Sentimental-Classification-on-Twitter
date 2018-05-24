
# coding: utf-8
import tweepy
from tweepy import OAuthHandler
import re 
import csv
#set consumer_key,consumer_secret,access_token,access_secret
consumer_key = 'your consumer_key'
consumer_secret = 'your consumer_secret'
access_token = 'your access_token'
access_secret = 'your access_secret'
auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
#set Twitter API. Let the API wait if If it reach the rate limit
api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
print("API setted up")
#path to save
path='/SOT/'

#Get positive tweets with searching keyword 'happy' and extract most recent 2000 tweets
happy_tweet=tweepy.Cursor(api.search,q='happy').items(2000)
happy_out_tweet=[[tweet.id_str, tweet.text,
             tweet.user.id, 
             tweet.user.location,tweet.coordinates,tweet.place,
             tweet.favorite_count,tweet.retweet_count, tweet.source,
             tweet.in_reply_to_status_id_str,tweet.in_reply_to_user_id_str,
             ] for tweet in happy_tweet]

# save the data as csv
with open(path+'happy.csv', 'w', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tweet.id_str", "tweet.text",
                     "tweet.user.screen_name", 
                     "tweet.user.location","tweet.coordinates","tweet.place",
                     "tweet.favorite_count","tweet.retweet_count", "tweet.source",
                     "tweet.coordinates"
                     ])
    writer.writerows(happy_out_tweet)
print('happy Data saved')

#Get negative tweets with searching keyword 'sad' and extract most recent 2000 tweets
sad_tweet=tweepy.Cursor(api.search,q='sad').items(2000)
sad_out_tweet=[[tweet.id_str, tweet.text,
             tweet.user.id, 
             tweet.user.location,tweet.coordinates,tweet.place,
             tweet.favorite_count,tweet.retweet_count, tweet.source,
             tweet.in_reply_to_status_id_str,tweet.in_reply_to_user_id_str,
             ] for tweet in sad_tweet]

# save the data as csv
with open(path+'sad.csv', 'w', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tweet.id_str", "tweet.text",
                     "tweet.user.screen_name", 
                     "tweet.user.location","tweet.coordinates","tweet.place",
                     "tweet.favorite_count","tweet.retweet_count", "tweet.source",
                     "tweet.coordinates"
                     ])
    writer.writerows(sad_out_tweet)
print('sad Data saved')

