import re
import pandas as pd
import warnings
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

def remove_pattern(txt,pattern):
    txt = str(txt)
    return " ".join(filter(lambda x: x[0] != pattern, txt.split()))

train  = pd.read_csv('train-data-with-headers.csv',encoding='latin-1',index_col=False,low_memory=False)
test = pd.read_csv('test-data.csv')

#combi = train.append(test, ignore_index=True)
train['tidy_tweet'] = [remove_pattern(x,'@') for x in train['tweet']]
print(train['tidy_tweet'].head(10))
train['tidy_tweet'] = train['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
train['tidy_tweet'] = [remove_pattern(x,'#') for x in train['tidy_tweet']]
print(train['tidy_tweet'].head(10))
train['tidy_tweet'] = train['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

# train['tidy_tweet'] = np.vectorize(remove_pattern)(train['tweet'], '@[\w]*')
print(train['tidy_tweet'].head(10))
