from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sqlite3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from unidecode import unidecode
import time
import preprocessor as p
import sqlite3

analyzer = SentimentIntensityAnalyzer()
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.NUMBER)


conn = sqlite3.connect('twitter_temp.db')
c = conn.cursor()

def create_table():
    try:
        c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, tweet TEXT, tweet_trans TEXT, location TEXT, sentiment REAL)")
        c.execute("CREATE INDEX fast_unix ON sentiment(unix)")
        c.execute("CREATE INDEX fast_tweet ON sentiment(tweet)")
        #c.execute("CREATE INDEX fast_tweet_trans ON sentiment(tweet_trans)")
        c.execute("CREATE INDEX fast_sentiment ON sentiment(sentiment)")
        conn.commit()
    except Exception as e:
        print(str(e))
create_table()

ckey=######
csecret=######
atoken=######
asecret=######

class listener(StreamListener):

    def on_data(self, data):
        try:
            data = json.loads(data)
            tweet = unidecode(data['text'])
            tweet = p.clean(tweet)
            #print (data['user']['location'])
            try:
            	location = unidecode(data['user']['location'])
            except:
            	location = "NA"
            #print (location)
            time_ms = data['timestamp_ms']
            vs = analyzer.polarity_scores(tweet)
            sentiment = vs['compound']
            tweet_trans = tweet
            #print(tweet_trans)
            c.execute("INSERT INTO sentiment (unix, tweet,tweet_trans, location, sentiment) VALUES (?, ?, ?, ?, ?)",
                  (time_ms, tweet, tweet_trans, location, sentiment))
            conn.commit()

        except KeyError as e:
            print(str(e))
        return(True)

    def on_error(self, status):
        print(status)


while True:

    try:
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        twitterStream = Stream(auth, listener())
        twitterStream.filter(track=["a","e","i","o","u"])
    except Exception as e:
        print(str(e))
        time.sleep(5)

# import sqlite3
# conn = sqlite3.connect('twitter_new.db')
# c = conn.cursor()
# df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 1000", conn ,params=('%' + my_id + '%',))
