import pickle
import csv
from nltk.sentiment import vader
import Train
import numpy as np
import pandas as pd
import Tweet_Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_sentiment(text):

    if type(float) == type(text):
        return 0
    # sia = vader.SentimentIntensityAnalyzer()
    # return sia.polarity_scores(text)['compound']

    pickle_in = pickle.open('tweet.pickle','rb')
    model = pickle.load(pickle_in)


def main(fileName):
    preprocessedFileName = Tweet_Preprocessing.preprocess(fileName,'Text','utf-8')
    print('Data Preprocessed...')
    prediction_dataset = pd.read_csv(preprocessedFileName, encoding='Latin-1', index_col=False, usecols=range(13), low_memory=False)
    print('Tweet Preprocessed File Read...')
    train_dataset = pd.read_csv('preprocessed-train-data.csv', encoding='Latin-1', index_col=False, usecols=range(7), low_memory=False)
    print('Train Preprocessed File Read...')
    train_dataset.Tidy_Tweet = train_dataset.Tidy_Tweet.fillna(value="")
    prediction_dataset.Tidy_Tweet = prediction_dataset.Tidy_Tweet.fillna(value="")

    x = np.array(train_dataset.Tidy_Tweet)
    x_train,_ ,_ ,_ = train_test_split(x, y, test_size=0.2, random_state=42)
    data_train = x_train
    tfv = TfidfVectorizer(sublinear_tf=True, stop_words="english")
    _ = tfv.fit_transform(data_train)

    print("Training Model...")
    pickle_in = open('logistic.pickle', 'rb')
    model = pickle.load(pickle_in)

    x_prediction = np.array(prediction_dataset.Tidy_Tweet)
    features_x_prediction = tfv.fit_transform(x_prediction)

    print("Features Extracted...")
    prediction_dataset['Score'] = model.predict(features_x_prediction)
    prediction_dataset.to_csv('output_prediction.csv', index=False)
    print('Output File Generated')
    print('Sentimental Analysis Of Tweets Complete...')

    Train.main(preprocessedFileName)


if __name__ == "__main__":
    main('tweet-data.csv')