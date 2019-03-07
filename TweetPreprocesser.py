# -*- coding: latin-1 -*-

import pandas as pd
import numpy as np
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
# from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
	test['tidy_tweet'] = np.vectorize(remove_pattern)(test.loc[:, 'Text'], "http[\w]*")

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

	# print(tokenized_tweet)

	test['tidy_tweet'] = tokenized_tweet
	test['tidy_tweet'].replace('', np.nan,inplace=True)
	test.dropna(inplace=True)
	test['tidy_tweet'].to_csv('PreprocessedData/'+j+'-preprocessed.csv')
	print(j+' preprocessed')
	# print(test['tidy_tweet'])
	# print(test['tidy_tweet'][411])

	# count = 0
	# for i in test['tidy_tweet']:
	# 	test['tidy_tweet'][count] = re.sub(r"http\S+", "", i)
	# 	count += 1



	# print(test['tidy_tweet'][411])



# all_words = ' '.join([text for text in test['tidy_tweet']])
#
# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
#
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()