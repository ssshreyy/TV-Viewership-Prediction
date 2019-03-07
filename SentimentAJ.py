import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# f = open("train-data.csv", "w")
# writer = csv.DictWriter(f,fieldnames=["polarity", "id","date", "query","username", "tweet"])
# writer.writeheader()
# f.close()

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt


train  = pd.read_csv('train-data-with-headers.csv',encoding='latin-1',index_col=False,low_memory=False)
test = pd.read_csv('test-data.csv')

#combi = train.append(test, ignore_index=True)
train['tweet'] =list( map(str,train['tweet']))
print(type(train['tweet'][0]))

train['tidy_tweet'] = np.vectorize(remove_pattern)(train['tweet'], '@[\w]*')
print(train['tidy_tweet'].head(10))
