import tweepy
import csv
import pandas as pd
import os
import sys
from apscheduler.schedulers.blocking import BlockingScheduler


# import cgi

# form = cgi.FieldStorage()
# kw = form.getvalue('keyword')
# print("keyword")
# keyword = sys.argv[1]
# time = sys.argv[2]


#keyword = input("Please enter search keyword: ")

#twitter API keys
consumer_key = ######
consumer_secret = ######
access_token = ######
access_secret = ######

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

states = dict()
temp = dict()

def fetch_hot_topics(state_name, state_code):
    try:
        trends = api.trends_place(state_code)
        #print(trends)
        for trend in trends[0]['trends']:
            states[state_name].append(trend['name'])

    except:
        #print(states)
        states[state_name].append("no result")

    states[state_name] = states[state_name][:10]

def main_func():
#read states
    global states, temp
    with open('states_woeid.csv','r') as f:
        lines = f.readlines()
        f.close()
    for line in lines:
        words = line.split(",")
        states[words[0]] = []
        temp[words[0]] = words[1].rstrip()
    #print(states)

    # trends = api.trends_closest(43.075968,-107.290284)
    # trends = api.trends_place(2391279)
    # print (trends)



    for key,value in temp.items():
        #print (key,value)
        fetch_hot_topics(key, value)

    #             states['Wyoming'] += 1

    outputFile = open('trends_data.csv', 'w')
    csvWriter = csv.writer(outputFile)
    csvWriter.writerow(['state','trends'])

    for key, val in states.items():
        csvWriter.writerow([key, val])
    states = dict()
    temp = dict()
    print('Successfully fetched trends data.')


scheduler = BlockingScheduler()
scheduler.add_job(main_func, 'interval', minutes=5)
scheduler.start()
