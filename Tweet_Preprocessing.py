import numpy as np
import warnings
import pandas as pd
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import LabeledSentence
from nltk.stem.porter import *
warnings.filterwarnings("ignore", category=DeprecationWarning)


def remove_http(txt):
    txt = str(txt)
    lst = list()
    for x in txt.split():
        if not x.startswith('http'):
            lst.append(x)
    return " ".join(lst)


def remove_pattern(txt,pattern):
    txt = str(txt)
    return " ".join(filter(lambda x: x[0] != pattern, txt.split()))


def preprocess(fileName,columnName,encode):

    train = pd.read_csv(fileName, encoding=encode, index_col=False, low_memory=False, usecols=range(12))
    print("File Read Successful...")

    # Remove @handle
    train['Tidy_Tweet'] = [remove_pattern(x,'@') for x in train[columnName]]
    print("Removed @handle...")

    #Remove URLs
    train['Tidy_Tweet'] = [remove_http(x) for x in train['Tidy_Tweet']]
    print("Removed URLs...")

    # Remove special characters, numbers, punctuations
    train['Tidy_Tweet'] = train['Tidy_Tweet'].str.replace("[^a-zA-Z#]", " ")
    print("Removed special characters, numbers, punctuations...")

    # Remove Short Words
    train['Tidy_Tweet'] = train['Tidy_Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    print("Removed short words...")

    # Tokenization
    tokenized_tweet_train = train['Tidy_Tweet'].apply(lambda x : x.split())
    print("Tokenization done...")

    # Stemming
    stemmer = PorterStemmer()
    tokenized_tweet_train = tokenized_tweet_train.apply(lambda x: [stemmer.stem(i) for i in x])
    print("Stemming done...")

    # Lammatization
    lemmatizer = WordNetLemmatizer()
    tokenized_tweet_train = tokenized_tweet_train.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    print("Lammatization done...")

    for i in range(len(tokenized_tweet_train)):
        tokenized_tweet_train[i] = ' '.join(tokenized_tweet_train[i])

    train['Tidy_Tweet'] = tokenized_tweet_train

    train.to_csv('tweet-preprocessed.csv', index=False)
    print("Output file generated...")
    return 'tweet-preprocessed.csv'