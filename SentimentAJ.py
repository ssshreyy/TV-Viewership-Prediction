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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier

# csv_input = pd.read_csv('input.csv')
# csv_input['Berries'] = csv_input['Name']
# csv_input.to_csv('output.csv', index=False)


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


# def main():

# f = open("train-data.csv", "w")
# writer = csv.DictWriter(f,fieldnames=["polarity", "id","date", "query","username", "tweet"])
# writer.writeheader()
# f.close()

train  = pd.read_csv('train-data-with-headers.csv',encoding='Latin-1',index_col=False,low_memory=False)
print("Train file read...")
test = pd.read_csv('test-data-with-headers.csv')
print("Test file read...")
#combi = train.append(test, ignore_index=True)

# Remove @handle
train['tidy_tweet'] = [remove_pattern(x,'@') for x in train['tweet']]
test['tidy_tweet'] = [remove_pattern(x,'@') for x in test['tweet']]

print("Removed @handle...")


#Remove URLs
train['tidy_tweet'] = [remove_http(x) for x in train['tweet']]
test['tidy_tweet'] = [remove_http(x) for x in test['tweet']]
# train['tidy_tweet'] = np.vectorize(remove_http)(train['tidy_tweet'], "http")
# test['tidy_tweet'] = np.vectorize(remove_http)(test['tidy_tweet'], "http")
print("Removed URLs...")

# Remove special characters, numbers, punctuations
train['tidy_tweet'] = train['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
test['tidy_tweet'] = test['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
print("Removed special characters, numbers, punctuations...")

# Remove Short Words
train['tidy_tweet'] = train['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
test['tidy_tweet'] = test['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
print("Removed short words...")

# Tokenization
tokenized_tweet_train = train['tidy_tweet'].apply(lambda x : x.split())
tokenized_tweet_test = test['tidy_tweet'].apply(lambda x : x.split())
print("Tokenization done...")


# Stemming
stemmer = PorterStemmer()
tokenized_tweet_train = tokenized_tweet_train.apply(lambda x: [stemmer.stem(i) for i in x])
tokenized_tweet_test = tokenized_tweet_test.apply(lambda x: [stemmer.stem(i) for i in x])
print("Stemming done...")


# Lammatization
lemmatizer = WordNetLemmatizer()
tokenized_tweet_train = tokenized_tweet_train.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
tokenized_tweet_test = tokenized_tweet_test.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
print("Lammatization done...")

for i in range(len(tokenized_tweet_train)):
    tokenized_tweet_train[i] = ' '.join(tokenized_tweet_train[i])

train['tidy_tweet'] = tokenized_tweet_train


for i in range(len(tokenized_tweet_test)):
    tokenized_tweet_test[i] = ' '.join(tokenized_tweet_test[i])

test['tidy_tweet'] = tokenized_tweet_test

train.to_csv('output_train.csv', index=False)
test.to_csv('output_test.csv', index=False)
print("Output files generated...")


# Wordcloud of most frequent words in train data
all_words = ' '.join([text for text in train['tidy_tweet']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

print("Wordcloud of most frequent words")
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
# Wordcloud of words in non racist/sexist tweet
normal_words =' '.join([text for text in train['tidy_tweet'][train['polarity'] == '0']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
print("Wordcloud of words in non racist/sexist tweet")

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Wordcloud of words in racist/sexist tweet
# negative_words = ' '.join([text for text in train['tidy_tweet'][train['polarity'] == '2']])
# wordcloud = WordCloud(width=800, height=500,random_state=21, max_font_size=110).generate(negative_words)
# print("Wordcloud of words in racist/sexist tweet")
#
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()

# Extracting hashtags from non racist/sexist tweets
print("Extracting hashtags from non racist/sexist tweets")
HT_regular = hashtag_extract(train['tidy_tweet'][train['polarity'] == '0'])

# Extracting hashtags from racist/sexist tweets
print("Extracting hashtags from racist/sexist tweets")
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


#Model building


train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting data into training and validation set
print("splitting data into training and validation set")
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['polarity'],
                                                          random_state=42,
                                                          test_size=0.3)

train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]
xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]
train_w2v = wordvec_df.iloc[:31962,:]
test_w2v = wordvec_df.iloc[31962:,:]
xtrain_w2v = train_w2v.iloc[ytrain.index,:]
xvalid_w2v = train_w2v.iloc[yvalid.index,:]
train_d2v = docvec_df.iloc[:31962,:]
test_d2v = docvec_df.iloc[31962:,:]
xtrain_d2v = train_d2v.iloc[ytrain.index,:]
xvalid_d2v = train_d2v.iloc[yvalid.index,:]
print("Entering random forest")
#Random Forest Model
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_bow, ytrain)

prediction = rf.predict(xvalid_bow)
print("F1 score is(prediction): ")
print(f1_score(yvalid, prediction))
test_pred = rf.predict(test_bow)
test['polarity'] = test_pred
submission = test[['id','polarity']]
submission.to_csv('sub_rf_bow.csv', index=False)
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_tfidf, ytrain)
prediction = rf.predict(xvalid_tfidf)
print("Some other F1 scores")
print(f1_score(yvalid, prediction))
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_w2v, ytrain)
prediction = rf.predict(xvalid_w2v)
print(f1_score(yvalid, prediction))
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_d2v, ytrain)
prediction = rf.predict(xvalid_d2v)
print(f1_score(yvalid, prediction))















# xgb_model = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_bow, ytrain)
# prediction = xgb_model.predict(xvalid_bow)
# f1_score(yvalid, prediction)
#
# test_pred = xgb_model.predict(test_bow)
# test['polarity'] = test_pred
# submission = test[['id','polarity']]
# submission.to_csv('sub_xgb_bow.csv', index=False)
#
#
# xgb = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_tfidf, ytrain)
#
# prediction = xgb.predict(xvalid_tfidf)
# f1_score(yvalid, prediction)
#
# xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_w2v, ytrain)
# prediction = xgb.predict(xvalid_w2v)
# f1_score(yvalid, prediction)
# xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_d2v, ytrain)
#
# prediction = xgb.predict(xvalid_d2v)
# f1_score(yvalid, prediction)
#
#
#
#
#
#
#
#
#
#
#
#
#
#

#
# #Model finetuning
#
# dtrain = xgb.DMatrix(xtrain_w2v, label=ytrain)
# dvalid = xgb.DMatrix(xvalid_w2v, label=yvalid)
# dtest = xgb.DMatrix(test_w2v)
#
#
# # Parameters that we are going to tune
# params = {
#     'objective':'binary:logistic',
#     'max_depth':6,
#     'min_child_weight': 1,
#     'eta':.3,
#     'subsample': 1,
#     'colsample_bytree': 1
# }
#
# def custom_eval(preds, dtrain):
#     labels = dtrain.get_label().astype(np.int)
#     preds = (preds >= 0.3).astype(np.int)
#     return [('f1_score', f1_score(labels, preds))]
#
#
# gridsearch_params = [
#     (max_depth, min_child_weight)
#     for max_depth in range(6,10)
#     for min_child_weight in range(5,8)
# ]
#
# max_f1 = 0. # initializing with 0
# best_params = None
# for max_depth, min_child_weight in gridsearch_params:
#     print("CV with max_depth={}, min_child_weight={}".format(
#                              max_depth,
#                              min_child_weight))
#
#     # Update our parameters
#     params['max_depth'] = max_depth
#     params['min_child_weight'] = min_child_weight
#
#     # Cross-validation
#     cv_results = xgb.cv(
#         params,
#         dtrain,
#         feval= custom_eval,
#         num_boost_round=200,
#         maximize=True,
#         seed=16,
#         nfold=5,
#         early_stopping_rounds=10
#     )
#
#     # Finding best F1 Score
#     mean_f1 = cv_results['test-f1_score-mean'].max()
#     boost_rounds = cv_results['test-f1_score-mean'].argmax()
#     print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))
#     if mean_f1 > max_f1:
#         max_f1 = mean_f1
#         best_params = (max_depth,min_child_weight)
#
# print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))
#
# params['max_depth'] = 8
# params['min_child_weight'] = 6
#
#
# gridsearch_params = [
#     (subsample, colsample)
#     for subsample in [i/10. for i in range(5,10)]
#     for colsample in [i/10. for i in range(5,10)]
# ]
#
# max_f1 = 0.
# best_params = None
# for subsample, colsample in gridsearch_params:
#     print("CV with subsample={}, colsample={}".format(
#                              subsample,
#                              colsample))
#
#     # Update our parameters
#     params['colsample'] = colsample
#     params['subsample'] = subsample
#
#     cv_results = xgb.cv(
#         params,
#         dtrain,
#         feval= custom_eval,
#         num_boost_round=200,
#         maximize=True,
#         seed=16,
#         nfold=5,
#         early_stopping_rounds=10
#     )
#
#     # Finding best F1 Score
#     mean_f1 = cv_results['test-f1_score-mean'].max()
#     boost_rounds = cv_results['test-f1_score-mean'].argmax()
#     print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))
#     if mean_f1 > max_f1:
#         max_f1 = mean_f1
#         best_params = (subsample, colsample)
#
# print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))
#
# params['subsample'] = .9
# params['colsample_bytree'] = .5
#
# max_f1 = 0.
# best_params = None
# for eta in [.3, .2, .1, .05, .01, .005]:
#     print("CV with eta={}".format(eta))
#
#     # Update ETA
#     params['eta'] = eta
#
#     # Run CV
#     cv_results = xgb.cv(
#         params,
#         dtrain,
#         feval= custom_eval,
#         num_boost_round=1000,
#         maximize=True,
#         seed=16,
#         nfold=5,
#         early_stopping_rounds=20
#     )
#
#     # Finding best F1 Score
#     mean_f1 = cv_results['test-f1_score-mean'].max()
#     boost_rounds = cv_results['test-f1_score-mean'].argmax()
#     print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))
#     if mean_f1 > max_f1:
#         max_f1 = mean_f1
#         best_params = eta
#
# print("Best params: {}, F1 Score: {}".format(best_params, max_f1))
#
# params['eta'] = .1
#
# max_f1 = 0.
# best_params = None
# for gamma in range(0,15):
#     print("CV with gamma={}".format(gamma/10.))
#
#     # Update ETA
#     params['gamma'] = gamma/10.
#
#     # Run CV
#     cv_results = xgb.cv(
#         params,
#         dtrain,
#         feval= custom_eval,
#         num_boost_round=200,
#         maximize=True,
#         seed=16,
#         nfold=5,
#         early_stopping_rounds=10
#     )
#
#     # Finding best F1 Score
#     mean_f1 = cv_results['test-f1_score-mean'].max()
#     boost_rounds = cv_results['test-f1_score-mean'].argmax()
#     print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))
#     if mean_f1 > max_f1:
#         max_f1 = mean_f1
#         best_params = gamma/10.
#
# print("Best params: {}, F1 Score: {}".format(best_params, max_f1))
#
# params['gamma'] = 1.2
#
#
# xgb_model = xgb.train(
#     params,
#     dtrain,
#     feval= custom_eval,
#     num_boost_round= 1000,
#     maximize=True,
#     evals=[(dvalid, "Validation")],
#     early_stopping_rounds=10
# )
#
# test_pred = xgb_model.predict(dtest)
# test['polarity'] = (test_pred >= 0.3).astype(np.int)
# submission = test[['id','polarity']]
# submission.to_csv('sub_xgb_w2v_06062018.csv', index=False)