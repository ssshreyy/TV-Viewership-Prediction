import re
from nltk.stem import WordNetLemmatizer
import numpy as np
from wordcloud import WordCloud
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from nltk.stem.porter import *

warnings.filterwarnings("ignore", category=DeprecationWarning)

# f = open("train-data.csv", "w")
# writer = csv.DictWriter(f,fieldnames=["polarity", "id","date", "query","username", "tweet"])
# writer.writeheader()
# f.close()

# def remove_pattern(input_txt, pattern):
#     r = re.findall(pattern, input_txt)
#     for i in r:
#         input_txt = re.sub(i, '', input_txt)
#
#     return input_txt

# def remove_http(input_txt, pattern):
# 	r = re.findall(pattern, input_txt)
# 	for i in r:
# 		input_txt = re.sub(i, '', input_txt)
#
# 	return input_txt

def remove_http(txt,pattern):

    return " ".join(filter(lambda x: x.startswith(pattern) != '', txt.split()))


def remove_pattern(txt,pattern):
    txt = str(txt)
    return " ".join(filter(lambda x: x[0] != pattern, txt.split()))

train  = pd.read_csv('test-data.csv',encoding='utf-8',index_col=False,low_memory=False)

# test = pd.read_csv('test-data.csv')

#combi = train.append(test, ignore_index=True)

train['tidy_tweet'] = [remove_pattern(x,'@') for x in train['tweet']]

train['tidy_tweet'] = np.vectorize(remove_http)(train['tidy_tweet'], "http")

train['tidy_tweet'] = train['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

train['tidy_tweet'] = [remove_pattern(x,'#') for x in train['tidy_tweet']]

train['tidy_tweet'] = train['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

tokenized_tweet = train['tidy_tweet'].apply(lambda x : x.split())
print(tokenized_tweet.head())


stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

print(tokenized_tweet.head())

lemmatizer = WordNetLemmatizer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

print(tokenized_tweet.head())
# train['tidy_tweet'] = np.vectorize(remove_pattern)(train['tweet'], '@[\w]*')
print(tokenized_tweet[0])
# all_words = ' '.join([text for text in train['tidy_tweet']])

# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()