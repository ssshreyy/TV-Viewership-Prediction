import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import csv


data = pd.read_csv("tweet.csv", error_bad_lines=False).replace('"', ' ', regex=True)
# print(data.columns)
print(type(data))
#text = data[['Text']].copy()
# print(type(text))
# print(text.columns)
# text.to_csv("outputFile.csv", sep='\t', encoding='utf-8')
# print(text.head(10))
# print(te)
with open("tweet.csv", 'r', encoding = "utf8") as csvfile:
    reader = csv.reader(csvfile)
    print(reader)

    for txt in reader:
        print(txt)
        txt = txt[7]
        txt = txt.replace('"', ' ')
        def clean_tweet(txt):
            return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", txt).split())


        # Sentiment Analysis
        # print(s.text.head(5))

        # Utility function to clean the text in a tweet by removing
        # links and special characters using regex.

        def analize_sentiment(txt):
            # '''
            # Utility function to classify the polarity of a tweet
            # using textblob.
            # '''
            analysis = TextBlob(clean_tweet(txt))
            if analysis.sentiment.polarity > 0:
                return 1
            elif analysis.sentiment.polarity == 0:
                return 0
            else:
                return -1


        # data['sa'] = analize_sentiment(txt)
        sa = analize_sentiment(txt)
        print(sa)
        # print(data['sa'])
        # sa = data[['sa']].copy()
        # sa.to_csv("tweet.csv", sep='\t', encoding='utf-8')
