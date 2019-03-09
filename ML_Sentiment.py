import csv
from nltk.sentiment import vader
import Train
import pandas as pd
import Tweet_Preprocessing

def calculate_sentiment(text):

    sia = vader.SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']


def main(fileName):
    preprocessedFileName = Tweet_Preprocessing.preprocess(fileName,'Text','utf-8')
    data = pd.read_csv(preprocessedFileName, low_memory=False, index_col=False, usecols=range(13))
    score = []
    count = 1
    for i in data['tidy_tweet']:
        # print(count)
        # count += 1
        score.append(calculate_sentiment(i))

    data['Score'] = score
    data.to_csv(preprocessedFileName,index = False)
    print('Sentimental Analysis Of Tweets Complete')
    Train.main(preprocessedFileName)


if __name__ == "__main__":
    main('tweet-data.csv')