# -*- coding: utf-8 -*-
import sys, getopt, datetime, codecs, json
import pandas as pd
from googletrans import Translator
from nltk.sentiment import vader

if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

def main(argv):
    if len(argv) == 0:
        print('You must pass some parameters.')
        return

    try:
        opts, _ = getopt.getopt(argv, "", (
        "username=", "near=", "within=", "since=", "until=", "querysearch=", "toptweets", "maxtweets=", "output="))

        tweetCriteria = got.manager.TweetCriteria()
        outputFileName = "tweet.csv"

        for opt, arg in opts:
            if opt == '--username':
                tweetCriteria.username = arg

            elif opt == '--since':
                tweetCriteria.since = arg

            elif opt == '--until':
                tweetCriteria.until = arg

            elif opt == '--querysearch':
                tweetCriteria.querySearch = arg

            elif opt == '--toptweets':
                tweetCriteria.topTweets = True

            elif opt == '--maxtweets':
                tweetCriteria.maxTweets = int(arg)

            elif opt == '--near':
                tweetCriteria.near = '"' + arg + '"'

            elif opt == '--within':
                tweetCriteria.within = '"' + arg + '"'

            elif opt == '--output':
                outputFileName = arg

        outputFile = codecs.open(outputFileName, "w+", "utf-8")

        outputFile.write('ID,Username,Author ID,Date,Time,Retweets,Favorites,Text,Mentions,Hashtags,Permalink,URL')

        print('Searching...\n')
        # sia = vader.SentimentIntensityAnalyzer()
        translator = Translator()

        def receiveBuffer(tweetss):
            for t in tweetss:
                s = translator.translate(t.text)
                outputFile.write(('\n%s,%s,%s,%s,%s,%d,%d,"""%s""",%s,%s,%s,%s' % (t.id, t.username, t.author_id, t.date.strftime("%Y-%m-%d"), t.date.strftime("%H:%M"), t.retweets, t.favorites, t.text, t.mentions, t.hashtags, t.permalink, t.urls)))
            # outputFile.write('%s' % (sia.polarity_scores(t.text)))
            outputFile.flush()
            print('More %d saved on file...\n' % len(tweetss))

        got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)

    except Exception:
        print('Arguments parser error, try -h ' + arg)
    finally:
        outputFile.close()
        print('Done. Output file generated "%s".' % outputFileName)


if __name__ == '__main__':
    main(sys.argv[1:])
