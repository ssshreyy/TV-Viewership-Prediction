import csv
from nltk.sentiment import vader
import Train
import pandas as pd
import Tweet_Preprocessing

def calculate_sentiment(text):

    if type(float) == type(text):
        return 0
    sia = vader.SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']


def main(fileName):
    preprocessedFileName = Tweet_Preprocessing.preprocess(fileName,'Text','utf-8')
    print('Data preprocessed...')
    data = pd.read_csv(preprocessedFileName, encoding='utf-8', low_memory=False, index_col=False, usecols=range(13))
    score = list()
    for i in data['Tidy_Tweet']:
        score.append(calculate_sentiment(i))

    data['Score'] = score
    data.to_csv(preprocessedFileName,index = False)
    print('Sentimental Analysis Of Tweets Complete')
    Train.main(preprocessedFileName)


if __name__ == "__main__":
    main('tweet-data.csv')