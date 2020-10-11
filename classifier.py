from Data_Preprocessing import data_preprocessing
import sys, os
sys.path.append(os.pardir)
import math
import random
import _pickle as cPickle
import numpy as np
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.feature_extraction import stop_words
from nltk.corpus import stopwords
model = KeyedVectors.load_word2vec_format('C:\\Users\\Bowen Yang\\PycharmProjects\\Research_Task1\\GoogleNews-vectors-negative300.bin.gz', binary=True)


[lib, con, neutral] = cPickle.load(open('ibcData.pkl', 'rb'))
columns = ['review', 'side']
dict_library = []
side_index = {'Conservative':0,'Liberal':1}

count_global = 0
def retrieve_data(side, str_side):
    global dict_library
    for tree in side[0:]:
        phrase = []
        current = tree.get_words()
        current = data_preprocessing(current)
        for node in tree:
            if hasattr(node, 'label'):
                if(node.label != "Neutral"):
                    phrase.append([node.get_words(), side_index[node.label]])
        dict_library.append({"review": current, "label": str_side,"phrase":phrase})


def LR1(x_train,x_test,y_train,y_test):
    vect = CountVectorizer(ngram_range=(1, 1), stop_words='english')
    print("LR1 data train", len(x_train))
    # vect = CountVectorizer()
    x_train_review_bow = vect.fit_transform(x_train)
    x_test_review_bow = vect.transform(x_test)

    clf2 = LogisticRegression()
    clf2.fit(x_train_review_bow, y_train)
    # y_predtfr = clf2.predict(x_test_review_tfidf)
    score = clf2.score(x_test_review_bow, y_test)
    # print('logistic regresion tfidf accuracy', accuracy_score(y_test,y_predtfr))
    print("score", score)

def LR2(phrase_train, x_test,y_test):
    x_phrase_train = []
    y_phrase_train = []
    for node in phrase_train:
        x_phrase_train.append(node[0])
        y_phrase_train.append(node[1])
    print("length of training for LR2", len(x_phrase_train))
    vect = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    x_train_review_bow = vect.fit_transform(x_phrase_train)
    x_test_review_bow = vect.transform(x_test)
    lr = LogisticRegression()
    lr.fit(x_train_review_bow, y_phrase_train)
    score = lr.score(x_test_review_bow, y_test)
    print("score", score)


def word2vec_LR(x_train, y_train, x_test, y_test):
    x_train_data_vec = word2vec(x_train)
    x_test_data_vec = word2vec(x_test)
    lr = LogisticRegression()
    lr.fit(x_train_data_vec, y_train)
    score = lr.score(x_test_data_vec, y_test)
    print("score", score)

#def matrix_add(a ,b):
#    result = []
 #   len = len(a)
 ### return result


def word2vec(arr):
    data_vec = []
    # tokenization
    for sentence in arr:
        line = []
        con = 1
        word_count = 0
        for word in sentence.split(" "):
            if word not in stopwords.words('english'):
                if (con == 1):
                    try:
                        line = model[word]
                        con = 0
                    except:
                        con = 1
                else:
                    try:
                        line = np.add(line, model[word])
                        word_count += 1
                    except:
                        print("useless word", word)
                print("current word count: ", word_count)
        line = np.true_divide(line, word_count)
        data_vec.append(line)
    return data_vec













total_x = []
total_y = []
x_train = []
y_train = []
x_test = []
y_test = []
phrase_total = []
#retrieve_data(neutral, 'neu')
retrieve_data(con, 0)
retrieve_data(lib, 1)
#df = pd.DataFrame(dict_library, columns=columns)
lim = int(math.floor(.75 * len(dict_library)))
random.shuffle(dict_library)
print('start')
count = 0
for data in dict_library:
    total_x.append(data["review"])
    total_y.append(data["label"])
    count += len(data["phrase"])
    for phrase in data["phrase"]:
        count = count+1
        print('append', count)
        phrase_total.append(phrase)
print(count)
print('start1')
lim_phrase = int(math.floor(.75 * len(phrase_total)))
phrase_train = phrase_total[:lim_phrase]
random.shuffle(phrase_train)
print('start2')

x_train = total_x[:lim]
x_test = total_x[lim:]
y_train = total_y[:lim]
y_test = total_y[lim:]
print("len of test", len(x_test))


#LR1(x_train, x_test, y_train, y_test)
#LR2(phrase_train,x_test,y_test)
print('run')
word2vec_LR(x_train, y_train, x_test ,y_test)





