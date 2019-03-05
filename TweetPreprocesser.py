# -*- coding: latin-1 -*-

import re
import pandas as pd
import numpy as np
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings


def remove_pattern(input_txt, pattern):
	r = re.findall(pattern, input_txt)
	for i in r:
		input_txt = re.sub(i, '', input_txt)

	return input_txt


warnings.filterwarnings("ignore", category=DeprecationWarning)

# train = pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, encoding='Latin-1')

# train = train.iloc[:, [0, 5]]
# print(train.columns)

dataFiles = ['tweet-2009', 'tweet-2010', 'tweet-2011', 'tweet-2012', 'tweet-2013', 'tweet-2014', 'tweet-2015']


for j in dataFiles:
	test = pd.read_csv(j+'.csv', encoding='Latin-1')
	test = test.loc[:, 'Text']

	# removing @handle
	test = pd.DataFrame(test)
	# print(train.columns)
	# train['tidy_tweet'] = np.vectorize(remove_pattern)(train.loc[:, 5], "@[\w]*")

	test['tidy_tweet'] = np.vectorize(remove_pattern)(test.loc[:, 'Text'], "@[\w]*")

	# remove special characters, numbers, punctuations
	# train['tidy_tweet'] = train['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
	test['tidy_tweet'] = test['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

	# Removing Short Words
	# train['tidy_tweet'] = train['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
	test['tidy_tweet'] = test['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
	# print(test.head(10))

	# Tokenization
	# tokenized_tweet_train = train['tidy_tweet'].apply(lambda x: x.split())
	tokenized_tweet = test['tidy_tweet'].apply(lambda x: x.split())

	# Stemming

	stemmer = PorterStemmer()

	# tokenized_tweet_train = tokenized_tweet_train.apply(lambda x: [stemmer.stem(i) for i in x])
	# tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

	#Lemmatizing

	lemmatizer = WordNetLemmatizer()
	tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

	for i in range(len(tokenized_tweet)):
		tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

	test['tidy_tweet'] = tokenized_tweet

	test['tidy_tweet'].to_csv(j+'-preprocessed.csv')