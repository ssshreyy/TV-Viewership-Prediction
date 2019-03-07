import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import nltk
import seaborn as sns
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from tqdm import tqdm
from gensim.models.doc2vec import LabeledSentence
warnings.filterwarnings("ignore", category=DeprecationWarning)

def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output

def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary

            continue
    if count != 0:
        vec /= count
    return vec

def remove_http(txt,pattern):

    return " ".join(filter(lambda x: x.startswith(pattern) != '', txt.split()))


def remove_pattern(txt,pattern):
    txt = str(txt)
    return " ".join(filter(lambda x: x[0] != pattern, txt.split()))


# def main():

# f = open("train-data.csv", "w")
# writer = csv.DictWriter(f,fieldnames=["polarity", "id","date", "query","username", "tweet"])
# writer.writeheader()
# f.close()

train  = pd.read_csv('train-data-with-headers.csv',encoding='Latin-1',index_col=False,low_memory=False)
# test = pd.read_csv('test-data.csv')
#combi = train.append(test, ignore_index=True)

# Remove @handle
train['tidy_tweet'] = [remove_pattern(x,'@') for x in train['tweet']]

# Remove URLs
#train['tidy_tweet'] = np.vectorize(remove_http)(train['tidy_tweet'], "http")

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


a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags
d = d.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=d, x="Hashtag", y="Count")
ax.set(ylabel='Count')
plt.show()

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

# selecting top 20 most frequent hashtags
e = e.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=e, x="Hashtag", y="Count")
plt.show()

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(train['tidy_tweet'])
bow.shape

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(train['tidy_tweet'])
tfidf.shape

tokenized_tweet = train['tidy_tweet'].apply(lambda x: x.split())  # tokenizing

model_w2v = gensim.models.Word2Vec(
    tokenized_tweet,
    size=200,  # desired no. of features/independent variables
    window=5,  # context window size
    min_count=2,
    sg=1,  # 1 for skip-gram model
    hs=0,
    negative=10,  # for negative sampling
    workers=2,  # no.of cores
    seed=34)

model_w2v.train(tokenized_tweet, total_examples=len(train['tidy_tweet']), epochs=20)

print(model_w2v.wv.most_similar(positive="tweet"))
#model_w2v.wv.most_similar(positive="trump")
#model_w2v['food']
print(len(model_w2v.wv.most_similar(positive="tweet")))

wordvec_arrays = np.zeros((len(tokenized_tweet), 200))

for i in range(len(tokenized_tweet)):
    wordvec_arrays[i, :] = word_vector(tokenized_tweet[i], 200)

wordvec_df = pd.DataFrame(wordvec_arrays)
print(wordvec_df.shape)

tqdm.pandas(desc="progress-bar")
labeled_tweets = add_label(tokenized_tweet)
print(labeled_tweets)

model_d2v = gensim.models.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model
                                  dm_mean=1, # dm = 1 for using mean of the context word vectors
                                  size=200, # no. of desired features
                                  window=5, # width of the context window
                                  negative=7, # if > 0 then negative sampling will be used
                                  min_count=5, # Ignores all words with total frequency lower than 2.
                                  workers=3, # no. of cores
                                  alpha=0.1, # learning rate
                                  seed = 23)

model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])

model_d2v.train(labeled_tweets, total_examples= len(train['tidy_tweet']), epochs=15)

docvec_arrays = np.zeros((len(tokenized_tweet), 200))

for i in range(len(train)):
    docvec_arrays[i, :] = model_d2v.docvecs[i].reshape((1, 200))

docvec_df = pd.DataFrame(docvec_arrays)
print(docvec_df.shape)


