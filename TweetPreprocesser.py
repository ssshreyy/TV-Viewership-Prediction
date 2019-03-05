# -*- coding: latin-1 -*-

import pandas as pd
import numpy as np
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer

import warnings


def remove_pattern(input_txt, pattern):
	r = re.findall(pattern, input_txt)
	for i in r:
		input_txt = re.sub(i, '', input_txt)

	return input_txt


warnings.filterwarnings("ignore", category=DeprecationWarning)

dataFiles = ['tweet-2009', 'tweet-2010', 'tweet-2011', 'tweet-2012', 'tweet-2013', 'tweet-2014', 'tweet-2015']


for j in dataFiles:
	test = pd.read_csv('TweetData/'+j+'.csv', encoding='Latin-1', low_memory=False)
	test = test.loc[:, 'Text']

	# removing @handle
	test = pd.DataFrame(test)

	test['tidy_tweet'] = np.vectorize(remove_pattern)(test.loc[:, 'Text'], "@[\w]*")

	# remove special characters, numbers, punctuations
	test['tidy_tweet'] = test['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

	# Removing Short Words
	test['tidy_tweet'] = test['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

	# Tokenization
	tokenized_tweet = test['tidy_tweet'].apply(lambda x: x.split())

	# Stemming

	# stemmer = PorterStemmer()
	# tokenized_tweet_train = tokenized_tweet_train.apply(lambda x: [stemmer.stem(i) for i in x])
	# tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

	#Lemmatizing

	lemmatizer = WordNetLemmatizer()
	tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

	for i in range(len(tokenized_tweet)):
		tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

	test['tidy_tweet'] = tokenized_tweet

	test['tidy_tweet'].to_csv('Preprocessing/'+j+'-preprocessed.csv')