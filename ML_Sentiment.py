import csv
from nltk.sentiment import vader
import Train
import pandas as pd

def calculate_sentiment(text):

    sia = vader.SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']


def main():
    data = pd.read_csv('tweet-sentiment.csv', low_memory=False, index_col=False, usecols=range(13))
    score = []
    count = 1
    for i in data['Text']:
        print(count)
        count += 1
        score.append(calculate_sentiment(i))

    data['Score'] = score
    data.to_csv("tweet-sentiment.csv",index = False)

    print('Sentimental Analysis Of Tweets Complete')
    Train.main()


if __name__ == "__main__":
    main()