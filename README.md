# FOMC minutes analysis and interest rate prediction

### Overall
Using NLP text analytics and machine learning to predict the interest rate change between two FOMC meetings

We use request and beautiful soup to download all the FOMC minutes from 1968 to 2019 and create different document-word matrix by different algorithm such as bow and tf-idf. 

Then we use machine learning to find a best model to predict the interest rate change direction (up or down) between two FOMC meetings interval, the result is quite promising

We then turn to some industry level data such as REIT index from 1977 to 2018, the result is even much better.

### What is the FOMC minutes and why is it important for interest rate prediction?

* Over time, the Fed has substantially increased its level of transparency thereby aiming at making monetary policy more effective. 
* The release of the minutes can have a sizable impact on Treasury bond yields. The impacts are largest when the tone of the minutes differs from the tone of the statement. This presumably leads markets to change their expectations of future monetary policy.
* The Fed is a highly predictable central bank and its communications have helped markets to anticipate future policy rate changes. The policy decision and communications by which the Fed or its officials explain monetary policy may have an impact on the market assessment of the future monetary policy course. 

### Scraping the FOMC website

We first use request and bs4 to download pdf version minutes from different links presented in the FOMC websites.

```Python
# get FOMC minutes from 1968 to 1992
from bs4 import BeautifulSoup
import requests
import re
import urllib.request
import os

base_url = "https://www.federalreserve.gov/monetarypolicy/fomchistorical"

transcript_links = {}
for year in range(1968, 1993):
  html_doc = requests.get(base_url + str(year) +'.htm') # get the link
  soup = BeautifulSoup(html_doc.content, 'html.parser') # extra the content
  links = soup.find_all("a", string=re.compile('Minutes*')) # find all links in the content
  print(links)
  link_base_url = "https://www.federalreserve.gov"
  transcript_links[str(year)] = [link_base_url + link["href"] for link in links] # store all links in each year
  print("Year Complete: ", year)
  
for year in transcript_links.keys():
    if not os.path.exists("./feddata/" + year):
        os.makedirs("./feddata/" + year)
    for link in transcript_links[year]:
        response = urllib.request.urlopen(str(link))
        name = re.search("[^/]*$", str(link))
        print(link)
        with open("./feddata/" + year + "/" + name.group(), 'wb') as f:
            f.write(response.read())
        print("file uploaded")
```
There are different format in different years, so we use different code to deal with minutes accordingly.

### Converting the pdf minutes to txt documents

the files we get from the websites are all pdf version which cannot be read directly, so we use a package called pdfminer to convert

```Python
# Converting PDFs to .txt files using the pdfminer in Python
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
import sys, getopt

# Convert pdf, returns its text content as a string
def convert(fname, pages=None): # The convert() function returns the text content of a PDF as a string.
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    # Create a string format object.
    output = StringIO()
    # Create a PDF resource manager object that stores shared resources.
    manager = PDFResourceManager()
    # Create a converter object.
    converter = TextConverter(manager, output, laparams=LAParams())
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text 
 
# Convert all pdfs in directory pdfDir, saves all resulting txt files to txtdir
def convertMultiple(pdfDir, txtDir):
    if pdfDir == "": pdfDir = os.getcwd() + "\\" # if no pdfDir passed in 
    for pdf in os.listdir(pdfDir): # iterate through pdfs in pdf directory
        fileExtension = pdf.split(".")[-1]
        if fileExtension == "pdf":
            pdfFilename = os.path.join(pdfDir, pdf) 
            text = convert(pdfFilename) # get string of text content of pdf
            textFilename = txtDir + pdf + ".txt" # create a .txt file
            textFile = open(textFilename, "w", encoding='utf-8') # open the.txt file
            textFile.write(text) # write the text content to the .txt file
			#textFile.close

src = r'C:\Users\Comete\feddata'
src_files = os.listdir(src)
print(src_files)
for file_name in src_files[18:-1]:
	pdfDir = os.path.join(src, file_name)
	txtDir = os.path.join(src, file_name)
	convertMultiple(pdfDir, txtDir)
```

### Prepocessing the minutes

Because of the conversion from pdf, some texts have been concatenated or carbled, we use re to replace all carbled characters and viterbi algorithm to seperate words.

```Python
##single minutes interval, signTerms for tfidf
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


```

### Removing less frequent words and scaling the documents

Then we use different packages of nlp and text analytics to create document-word matrix for furthur analysis, then remove the words that shows up in few docs and scale every document

```Python


### Step 4: Data Preprocessing: bag of words, tf-idf, scaling the data, import interest rate data, etc.
## Bag of Words with sklearn
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

# Scaling the BoW data onto one scale eliminating the sparsity 
bowScaled = preprocessing.scale(df_AnnualBow)
df_bowScaled = pd.DataFrame(bowScaled,columns=df_AnnualBow.columns)
minutes_BoW_sk2 = pd.concat([minutes,df_bowScaled],axis = 1)

## tf-idf with sklearn
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

# Scaling the tf-idf data onto one scale eliminating the sparsity 		
tfidfScaled = preprocessing.scale(df_Annualtfidf)
df_tfidfScaled = pd.DataFrame(tfidfScaled,columns=df_Annualtfidf.columns)
minutes_tfidf_sk = pd.concat([minutes,df_tfidfScaled],axis = 1)
```

### Calculating the interest rate change between two FOMC meetings and the correlation between words and FED rate change

We use FED rate daily data from website [**macrotrends**](https://www.macrotrends.net/2015/fed-funds-rate-historical-chart), and merge it with the FOMC minutes in the public release date(around 23 days later of the meeting date), then calculate the FED rate difference between two minutes.

After that, we calculate the top 20 words that are most correlated with FED rate change.

```Python

# import interest rate data and merge them
IR = pd.read_csv(r'C:\Users\Comete\Desktop\MFinRelated\nlp\NLPTA_project-master\NLPTA_project-master\fed-funds-rate-historical-chart.csv')
IR >> head(3)

minutes_BoW_sk2['oldDate'] =pd.to_datetime(minutes_BoW_sk2['file_name'],format='%Y%m%d',errors='ignore')
minutes_tfidf_sk['oldDate'] =pd.to_datetime(minutes_tfidf_sk['file_name'],format='%Y%m%d',errors='ignore')
IR['Date'] = pd.to_datetime(IR['date'],format='%Y/%m/%d',errors='ignore')

minutes_BoW_sk2['Date'] = minutes_BoW_sk2['oldDate'] + datetime.timedelta(days=23)
minutes_tfidf_sk['Date'] = minutes_tfidf_sk['oldDate'] + datetime.timedelta(days=23)

bow_IR = pd.merge(IR,minutes_BoW_sk2,on = 'Date',how = 'left')

tfIdf_IR = pd.merge(IR,minutes_tfidf_sk,on = 'Date',how = 'left')

tfIdf_IR_diff = bow_IR.dropna()
tfIdf_IR_diff['rateChange'] = tfIdf_IR_diff['fedRate'].shift(-1) - tfIdf_IR_diff['fedRate']
tfIdf_IR_diff = tfIdf_IR.dropna()
tfIdf_IR_diff['rateChange'] = tfIdf_IR_diff['fedRate'].shift(-1) - tfIdf_IR_diff['fedRate']

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
    plt.title(" correlations with IR change")
    plt.show()

tfIdf_IR_diff = tfIdf_IR_diff.dropna()
tfIdf_IR_diff=tfIdf_IR_diff.drop(['date_x','year','month','day','file_name','oldDate','content'],axis=1)
tfIdf_IR_diff.sort_values(by=['Date','rateChange'],ascending = True)
CorBowIR = CorTerms(tfIdf_IR_diff.columns[2:-1],tfIdf_IR_diff,tfIdf_IR_diff['rateChange'],top = 20,bottom = 20)
bow_top = CorBowIR[1]
bow_bottom = CorBowIR[2]
corBar(bow_top['keyterms'],bow_top['correlations'])
corBar(bow_bottom['keyterms'],bow_bottom['correlations'])

tfIdf_IR_diff = tfIdf_IR_diff.dropna()
tfIdf_IR_diff=tfIdf_IR_diff.drop(['date_x','year','month','day','file_name','oldDate','content'],axis=1)
tfIdf_IR_diff.sort_values(by=['Date','rateChange'],ascending = True)
CorTfidfIR = CorTerms(tfIdf_IR_diff.columns[2:-1],tfIdf_IR_diff,tfIdf_IR_diff['rateChange'],top = 20,bottom = 20)
tfIdf_top = CorTfidfIR[1]
tfIdf_bottom = CorTfidfIR[2]
corBar(tfIdf_top['keyterms'],tfIdf_top['correlations'])
corBar(tfIdf_bottom['keyterms'],tfIdf_bottom['correlations'])


```

### Logistic regression with FED rate moving direction

We first try to use OLS regression to predict the actual change of FED rate or use LSA to find out some interesting pattern in the documents, but none of them work well. 

So we turn our attention to predict the FED rate moving direction (up or down) by different machine learning algorithm, using words that are statistically significant in logistic regression.

```Python

# logistic regression

IR_ChID= np.where(tfIdf_IR_diff['rateChange']>0,1,0)
tfIdf_IR_diff.insert(1,'IR_ChID',IR_ChID)
tfIdf_IR_diff
## to improve the model, I would filter out thoes insignifiant terms
from sklearn.feature_selection import f_regression
words = tfIdf_IR_diff.columns[3:-1]
X = tfIdf_IR_diff[words] 
y = tfIdf_IR_diff['IR_ChID']
logisreg = f_regression(X, y, center=True)
Fvalue = logisreg[0]
Pvalue = logisreg[1]

stat_CortfidfIR = CorTfidfIR[0]
stat_CortfidfIR['Fvalue'] = Fvalue
stat_CortfidfIR['Pvalue'] = Pvalue

signTerms = stat_CortfidfIR.query('Pvalue < 0.05')

signTerms['Cor_P'] = signTerms['correlations'] /signTerms['Pvalue']*signTerms['Fvalue']
signTermsBottom = signTerms.sort_values(by='correlations', ascending=True) >> head(20)
signTermsTop = signTerms.sort_values(by = 'correlations', ascending=False) >> head(20)

corBar(signTermsTop['keyterms'],signTermsTop['correlations'])
corBar(signTermsBottom['keyterms'],signTermsBottom['correlations'])

ID_var = signTerms['keyterms'].tolist()
X = tfIdf_IR_diff[ID_var]
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
```

And we have tried different parameters and different method to increase our predictive power, the results are as below:

![best model results in different methods](https://raw.githubusercontent.com/Cometecomete/FOMC-minutes-analysis-and-interest-rate-prediction/master/best%20model%20results%20in%20different%20methods.png)

Then we turn to the real estate industry level data, the predictive power is even better as below (you can check all the code in our code file):

![best model results for real estate industry](https://raw.githubusercontent.com/Cometecomete/FOMC-minutes-analysis-and-interest-rate-prediction/master/best%20model%20results%20for%20real%20estate%20industry.png)

### Conclusion and others

We have done something else such as calculating the most correlated words in terms of FED rate change and get some meaningful words that have strong tendency, supporting our results to some extent. We think one possible explanation that our industry models work better is because that all the large companies in real estate industry pay a lot attention on the FOMC meetings and thus the FOMC minutes mean a lot to them, leading to a better correlation and prediction of the future performance in the industry.

We believe there still are so many improvements that could be done such as change the definition of the interest rate change and try to predict the moving direction after one month or six months after every minutes released; or use other industry data such as finance industry to see the predictive power; or simply improve the documents accuracy by manually checking.

But overall, we can conclude that using FOMC minutes we can have a relatively actual predictive power on FED interest rate change direction and relative industry performance.

Any comments or questions, please contact email: yuhang.xia@outlook.com

Thanks for reading!

![group symbol](https://raw.githubusercontent.com/Cometecomete/FOMC-minutes-analysis-and-interest-rate-prediction/master/4D-Intelli.png)


