import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from nltk.stem.porter import *
warnings.filterwarnings("ignore", category=DeprecationWarning)

def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


def remove_http(txt,pattern):

    return " ".join(filter(lambda x: x.startswith(pattern) != '', txt.split()))


def remove_pattern(txt,pattern):
    txt = str(txt)
    return " ".join(filter(lambda x: x[0] != pattern, txt.split()))


def main():

    # f = open("train-data.csv", "w")
    # writer = csv.DictWriter(f,fieldnames=["polarity", "id","date", "query","username", "tweet"])
    # writer.writeheader()
    # f.close()

    train  = pd.read_csv('test-data-with-headers.csv',encoding='utf-8',index_col=False,low_memory=False)
    # test = pd.read_csv('test-data.csv')
    #combi = train.append(test, ignore_index=True)

    # Remove @handle
    train['tidy_tweet'] = [remove_pattern(x,'@') for x in train['tweet']]

    # Remove URLs
    train['tidy_tweet'] = np.vectorize(remove_http)(train['tidy_tweet'], "http")

    # Remove special characters, numbers, punctuations
    train['tidy_tweet'] = train['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

    # Remove Short Words
    train['tidy_tweet'] = train['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

    # Tokenization
    tokenized_tweet = train['tidy_tweet'].apply(lambda x : x.split())

    # Stemming
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

    # Lammatization
    lemmatizer = WordNetLemmatizer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    train['tidy_tweet'] = tokenized_tweet

    # Wordcloud of most frequent words
    all_words = ' '.join([text for text in train['tidy_tweet']])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    # plt.figure(figsize=(10, 7))
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis('off')
    # plt.show()

    # Wordcloud of words in non racist/sexist tweet
    normal_words =' '.join([text for text in train['tidy_tweet'][train['polarity'] == '0']])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

    # Wordcloud of words in racist/sexist tweet
    negative_words = ' '.join([text for text in train['tidy_tweet'][train['polarity'] == '2']])
    wordcloud = WordCloud(width=800, height=500,random_state=21, max_font_size=110).generate(negative_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

    # Extracting hashtags from non racist/sexist tweets
    HT_regular = hashtag_extract(train['tidy_tweet'][train['polarity'] == '0'])

    # Extracting hashtags from racist/sexist tweets
    HT_negative = hashtag_extract(train['tidy_tweet'][train['polarity'] == '2'])

    # Unnesting list
    HT_regular = sum(HT_regular,[])
    HT_negative = sum(HT_negative,[])

if __name__ == '__main__':
    main()