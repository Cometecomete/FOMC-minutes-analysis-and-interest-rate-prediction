##single minutes interval, signTerms for bow, real estate
import os
from dfply import *
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Pool
import warnings
from nltk.stem import WordNetLemmatizer
warnings.filterwarnings("ignore",category=DeprecationWarning)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import statsmodels.api as sm
minutes = pd.read_csv(r'C:\Users\Comete\Desktop\MFinRelated\nlp\NLPTA_project-master\NLPTA_project-master\docs.csv')

from collections import Counter


def word_prob(word): 
    return dictionary[word]/total
	
	
def words(text): 
    return re.findall('[a-z]+', text.lower()) 
	
def viterbi_segment(text):
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]:i])
        i = lasts[i]
    words.reverse()
    return words, probs[-1]

dictionary = Counter(words(open(r'C:\Users\Comete\big.txt').read()))
max_word_length = max(map(len, dictionary))
total = float(sum(dictionary.values()))


# 1. data processing
minutes.pop(minutes.columns[0])
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])

stop_words = stopwords.words('english')

import datetime
Month = [datetime.date(2008, i, 1).strftime('%B').lower() for i in range(1,13)]
stop_words.extend(['year','month','day','mr','meeting','committee','ms','federal','page']
                    + Month)

import gensim
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

texts = list(sent_to_words(minutes['content']))

bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)

from gensim.utils import simple_preprocess
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in data:
        doc = nlp(re.sub('\_',''," ".join(sent)))
        tokens = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        new_tokens = []
        for i in tokens:
            for j in viterbi_segment(i)[0]:
                new_tokens.append(j)
        texts_out.append(new_tokens)
    return texts_out

corpus_no_stops = remove_stopwords(texts)
corpus_bigrams = make_bigrams(corpus_no_stops)
set(['infla_tion' in word for word in corpus_bigrams])
data_lemmatized = lemmatization(corpus_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

corpus = [' '.join(wordList) for wordList in data_lemmatized]
set(['infla_tion' in word for word in corpus])

# bag of words with sklearn
vectorizer = CountVectorizer(stop_words = 'english',lowercase = True)
AnnualBow = vectorizer.fit_transform(corpus)
df_AnnualBow = pd.DataFrame(AnnualBow.A,columns = vectorizer.get_feature_names())
for term in ['year','month','day']:
    try:
        df_AnnualBow.pop(term)
    except:
        continue

'month' in df_AnnualBow.columns

df_AnnualBow.astype(bool).sum(axis=1)
frequency = df_AnnualBow.astype(bool).sum(axis=0)
less = frequency[frequency<3]
df_AnnualBow=df_AnnualBow.drop(less.index,axis=1)

##remove thoes meaningness words

bowScaled = preprocessing.scale(df_AnnualBow)
df_bowScaled = pd.DataFrame(bowScaled,columns=df_AnnualBow.columns)
minutes_BoW_sk2 = pd.concat([minutes,df_bowScaled],axis = 1)

# tf-idf sklearn
v = TfidfVectorizer(stop_words='english', max_df=0.9)
tfidf = v.fit_transform(corpus)
df_Annualtfidf = pd.DataFrame(tfidf.A,columns = v.get_feature_names())

for term in ['year','month','day']:
    try:
        df_Annualtfidf.pop(term)
    except:
        continue

df_Annualtfidf.astype(bool).sum(axis=1)
frequency2 = df_Annualtfidf.astype(bool).sum(axis=0)
less2 = frequency2[frequency<3]
df_Annualtfidf=df_Annualtfidf.drop(less2.index,axis=1)
		
tfidfScaled = preprocessing.scale(df_Annualtfidf)
df_tfidfScaled = pd.DataFrame(tfidfScaled,columns=df_Annualtfidf.columns)
minutes_tfidf_sk = pd.concat([minutes,df_tfidfScaled],axis = 1)

# import interest rate data and merge them
IR = pd.read_csv(r'C:\Users\Comete\USREINDEX.csv')
IR >> head(3)

minutes_BoW_sk2['oldDate'] =pd.to_datetime(minutes_BoW_sk2['file_name'],format='%Y%m%d',errors='ignore')
minutes_tfidf_sk['oldDate'] =pd.to_datetime(minutes_tfidf_sk['file_name'],format='%Y%m%d',errors='ignore')
IR['Date'] = pd.to_datetime(IR['DATE'],format='%Y/%m/%d',errors='ignore')

minutes_BoW_sk2['Date'] = minutes_BoW_sk2['oldDate'] + datetime.timedelta(days=23)
minutes_tfidf_sk['Date'] = minutes_tfidf_sk['oldDate'] + datetime.timedelta(days=23)

bow_IR = pd.merge(IR,minutes_BoW_sk2,on = 'Date',how = 'left')

tfIdf_IR = pd.merge(IR,minutes_tfidf_sk,on = 'Date',how = 'left')

bow_IR_diff = bow_IR.dropna()
bow_IR_diff['rateChange'] = bow_IR_diff['WILLREITIND'].shift(-1) - bow_IR_diff['WILLREITIND']
tfIdf_IR_diff = tfIdf_IR.dropna()
tfIdf_IR_diff['rateChange'] = tfIdf_IR_diff['WILLREITIND'].shift(-1) - tfIdf_IR_diff['WILLREITIND']

def CorTerms(terms,df_sum,y,top = None,bottom = None):
    correlations = [np.corrcoef(y,df_sum[term])[0,1]
                    for term in list(terms)]## change the index
    IR_corTerms = pd.DataFrame({'keyterms':terms,'correlations':correlations})
    top = IR_corTerms.sort_values(by = 'correlations',ascending = False) >> head(top)
    bottom = IR_corTerms.sort_values(by = 'correlations',ascending = True) >> head(bottom)
    return IR_corTerms,top,bottom

def corBar(x,y):
    plt.barh(range(len(x)), y, height=0.7, color='steelblue', alpha=0.8) 
    plt.yticks(range(len(x)), x)
    plt.xlabel("correlations")
    plt.ylabel('keyterms')
    plt.title(" correlations with REIT change")
    plt.show()

bow_IR_diff = bow_IR_diff.dropna()
bow_IR_diff=bow_IR_diff.drop(['Unnamed: 0','Index','year','month','day','file_name','oldDate','content'],axis=1)
bow_IR_diff.sort_values(by=['Date','rateChange'],ascending = True)
CorBowIR = CorTerms(bow_IR_diff.columns[3:-1],bow_IR_diff,bow_IR_diff['rateChange'],top = 20,bottom = 20)
bow_top = CorBowIR[1]
bow_bottom = CorBowIR[2]
corBar(bow_top['keyterms'],bow_top['correlations'])
corBar(bow_bottom['keyterms'],bow_bottom['correlations'])

tfIdf_IR_diff = tfIdf_IR_diff.dropna()
tfIdf_IR_diff=tfIdf_IR_diff.drop(['Unnamed: 0','Index','year','month','day','file_name','oldDate','content'],axis=1)
tfIdf_IR_diff.sort_values(by=['Date','rateChange'],ascending = True)
CorTfidfIR = CorTerms(tfIdf_IR_diff.columns[3:-1],tfIdf_IR_diff,tfIdf_IR_diff['rateChange'],top = 20,bottom = 20)
tfIdf_top = CorTfidfIR[1]
tfIdf_bottom = CorTfidfIR[2]
corBar(tfIdf_top['keyterms'],tfIdf_top['correlations'])
corBar(tfIdf_bottom['keyterms'],tfIdf_bottom['correlations'])


# logistic regression

IR_ChID= np.where(bow_IR_diff['rateChange']>0,1,0)
bow_IR_diff.insert(1,'IR_ChID',IR_ChID)
bow_IR_diff
## to improve the model, I would filter out thoes insignifiant terms
from sklearn.feature_selection import f_regression
words = bow_IR_diff.columns[4:-1]
X = bow_IR_diff[words] 
y = bow_IR_diff['IR_ChID']
logisreg = f_regression(X, y, center=True)
Fvalue = logisreg[0]
Pvalue = logisreg[1]

stat_CorbowIR = CorBowIR[0]
stat_CorbowIR['Fvalue'] = Fvalue
stat_CorbowIR['Pvalue'] = Pvalue

signTerms = stat_CorbowIR.query('Pvalue < 0.05')

signTerms['Cor_P'] = signTerms['correlations'] /signTerms['Pvalue']*signTerms['Fvalue']
signTermsBottom = signTerms.sort_values(by='correlations', ascending=True) >> head(20)
signTermsTop = signTerms.sort_values(by = 'correlations', ascending=False) >> head(20)

corBar(signTermsTop['keyterms'],signTermsTop['correlations'])
corBar(signTermsBottom['keyterms'],signTermsBottom['correlations'])

ID_var = signTerms['keyterms'].tolist()
X = bow_IR_diff[ID_var]
y= IR_ChID


#machine learning for prediction

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import metrics
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn import preprocessing
from sklearn import utils


lab_enc = preprocessing.LabelEncoder()
y_train_encoded = lab_enc.fit_transform(y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifiers = {
    'SVR':svm.SVR(),
	'SVC':SVC(),
    'SGD':linear_model.SGDRegressor(),
    'BAYES':linear_model.BayesianRidge(),
    'LL':linear_model.LassoLars(),
    'ARD':linear_model.ARDRegression(),
    'PA':linear_model.PassiveAggressiveRegressor(),
    'TS':linear_model.TheilSenRegressor(),
    'L':linear_model.LinearRegression()
	}

train_scores = []
test_scores = []
names = []
models = {}
for key in classifiers.keys(): 
    clf = classifiers[key]
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    y_test_predict = clf.predict(X_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    names.append(key)

models['train_score'] = train_scores
models['test_score'] = test_scores
models['model'] = names
df_models = pd.DataFrame(models)
df_models.to_csv('models.csv')


model_names = []
best_scores = []
best_models = []

for key in classifiers.keys(): 
  try:
    clf = classifiers[key]
    parameters = {'kernel':('linear', 'rbf'), 'C':(0.1, 1,5, 10)}
    gs = GridSearchCV(clf, parameters)
    gs.fit(X_train, y_train)
    best_scores.append(gs.best_score_)
    best_models.append(gs.best_params_)
    model_names.append(key)
  except:
    continue

best = {}
best['name'] = model_names
best['score'] = best_scores
best['param'] = best_models
df_best = pd.DataFrame(best)
df_best.to_csv('best.csv')

parameters = {'kernel':('linear', 'rbf'), 'C':(0.1,0.5,0.8, 1,1.2,1.5,2,5, 10)}
gs = GridSearchCV(SVC(), parameters)
gs.fit(X_train, y_train)
df_gs=pd.DataFrame(gs.cv_results_)
df_gs.to_csv('best2.csv')
gs.score(X_test,y_test)
	