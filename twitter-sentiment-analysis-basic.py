import numpy as np
import pandas as pd
import re
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

def import_tweets(filename, header = None):
    #import data from csv file via pandas library
    tweet_dataset = pd.read_csv(filename,header = header, encoding='Latin-1',usecols=range(6), low_memory=False, index_col=False)
    #the column names are based on sentiment140 dataset provided on kaggle
    tweet_dataset.columns = ['sentiment','id','date','flag','user','text']
    #delete 3 columns: flags,id,user, as they are not required for analysi
    # for i in ['flag','id','user','date']: del tweet_dataset[i] # or tweet_dataset = tweet_dataset.drop(["id","user","date","user"], axis = 1)
    #in sentiment140 dataset, positive = 4, negative = 0; So we change positive to 1
    #tweet_dataset.sentiment = tweet_dataset.sentiment.replace(4,1)
    return tweet_dataset

def preprocess_tweet(tweet):
    #Preprocess the text in a single tweet
    #arguments: tweet = a single tweet in form of string
    #convert the tweet to lower case

    if type(float) == type(tweet):
        return '-'

    tweet.lower()
    #convert all urls to sting "URL "
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #convert all @username to "AT_USER "
    tweet = re.sub('@[^\s]+','AT_USER', tweet)
    #correct all multiple white spaces to a single white space
    tweet = re.sub('[\s]+', ' ', tweet)
    #convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet


def feature_extraction(data, method = "tfidf"):
    #arguments: data = all the tweets in the form of array, method = type of feature extracter
    #methods of feature extractions: "tfidf" and "doc2vec"
    if method == "tfidf":
        tfv=TfidfVectorizer(sublinear_tf=True, stop_words = "english") # we need to give proper stopwords list for better performance
        features=tfv.fit_transform(data)
    elif method == "doc2vec":
        None
    else:
        return "Incorrect inputs"
    return features

def train_classifier(features_train,features_test,label_train,label_test,classifier = "logistic_regression"):
    if classifier == "logistic_regression": # auc (train data): 0.8780618441250002
        model = LogisticRegression(C=1.)
    elif classifier == "naive_bayes": # auc (train data): 0.8767891829687501
        model = MultinomialNB()
    elif classifier == "svm": # can't use sklearn svm, as way too much of data so way to slow. have to use tensorflow for svm
        model = SVC()
    elif classifier == "random_forest":
        model=RandomForestClassifier(n_estimators=400, random_state=11)
    else:
        print("Incorrect selection of classifier")
    #fit model to data
    model.fit(features_train, label_train)
    print("Model fitting done...")
    with open('logistic.pickle', 'wb') as file:
        pickle.dump(model, file)
    accuracy=model.score(features_test,label_test)
    print("Accuracy is:")
    print(accuracy)
    #make prediction on the test data
    # probability_to_be_positive = model.predict_proba(features_test)[:,1]
    #chcek AUC(Area Undet the Roc Curve) to see how well the score discriminates between negative and positive
    #print ("auc (train data):" , roc_auc_score(label_test, probability_to_be_positive))
    #print top 10 scores as a sanity check
    #print ("top 10 scores: ", probability_to_be_positive[:10])
    return model

# #apply the preprocess function for all the tweets in the dataset
# tweet_dataset = import_tweets("1.csv")
# tweet_dataset.text = tweet_dataset.text.fillna(value="")
#
# #tweet_dataset = import_tweets("trainandtest.csv")
# #tweet_dataset = import_tweets("new.csv")
# print("File read")
#
# tweet_dataset['Tidy_Tweet'] = tweet_dataset['text'].apply(preprocess_tweet)
# print("Preprocessing done")
#
# tweet_dataset['Tidy_Tweet'] = tweet_dataset['Tidy_Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
# print("Removed short words...")
#
# # Tokenization
# tokenized_tweet_train = tweet_dataset['Tidy_Tweet'].apply(lambda x: x.split())
# print("Tokenization done...")
#
# # Stemming
# stemmer = PorterStemmer()
# tokenized_tweet_train = tokenized_tweet_train.apply(lambda x: [stemmer.stem(i) for i in x])
# print("Stemming done...")
#
# # Lammatization
# lemmatizer = WordNetLemmatizer()
# tokenized_tweet_train = tokenized_tweet_train.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
# print("Lammatization done...")
#
# for i in range(len(tokenized_tweet_train)):
#     tokenized_tweet_train[i] = ' '.join(tokenized_tweet_train[i])
#
# tweet_dataset['Tidy_Tweet'] = tokenized_tweet_train
# tweet_dataset.to_csv('preprocessed-train-data.csv',index = False)
# print('File generated')
tweet_dataset = pd.read_csv('preprocessed-train-data.csv', encoding='Latin-1', index_col=False, usecols=range(7), low_memory=False)
tweet_dataset.Tidy_Tweet = tweet_dataset.Tidy_Tweet.fillna(value="")
#
x = np.array(tweet_dataset.Tidy_Tweet)
y = np.array(tweet_dataset.sentiment)
#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
data_train  = x_train
label_train = y_train
data_test   = x_test
label_test  = y_test
#
print("Extracting features")
tfv = TfidfVectorizer(sublinear_tf=True,stop_words="english")  # we need to give proper stopwords list for better performance
features_train=tfv.fit_transform(data_train)
# features_test=tfv.transform(data_test)

print("Training")

#model=train_classifier(features_train,features_test, label_train,label_test, "logistic_regression")
pickle_in = open('logistic.pickle', 'rb')
model = pickle.load(pickle_in)

prediction_dataset = pd.read_csv('tweet-preprocessed.csv', encoding='Latin-1', index_col=False, usecols=range(13), low_memory=False)
prediction_dataset.Tidy_Tweet = prediction_dataset.Tidy_Tweet.fillna(value="")
x_prediction = np.array(prediction_dataset.Tidy_Tweet)
features_x_prediction=tfv.fit_transform(x_prediction)

prediction_dataset['Score'] = model.predict(features_x_prediction)
prediction_dataset.to_csv('output_prediction.csv', index=False)

