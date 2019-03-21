# FOMC minutes analysis and interest rate prediction
### Overall
Using NLP text analytics and machine learning to predict the interest rate change between two FOMC meetings

We use request and beautiful soup to download all the FOMC minutes from 1968 to 2019 and create different document-word matrix by different algorithm such as bow and tf-idf. 

Then we use machine learning to find a best model to predict the interest rate change direction (up or down) between two FOMC meetings interval, the result is quite promising

We then turn to some industry level data such as REIT index from 1977 to 2018, the result is even much better.

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

### prepocessing the minutes

Because of the conversion from pdf, some texts have been concatenated or carbled, we use re to replace all carbled characters and viterbi algorithm to seperate words.

```Python

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

def lemmatization(data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): # Do lemmatization keeping only noun, adj, vb, adv
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
```

### converting the documents into bow or tf-idf

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

### Calculating the interest rate change between two FOMC meetings

We use FED rate daily data from website [**macrotrends**](https://www.macrotrends.net/2015/fed-funds-rate-historical-chart), and merge it with the FOMC minutes in the public release date(around 23 days later of the meeting date), then calculate the FED rate difference between two minutes.
```Python

## Import interest rate data and merge them
IR = pd.read_csv(r'C:\Users\Comete\Desktop\MFinRelated\nlp\NLPTA_project-master\NLPTA_project-master\fed-funds-rate-historical-chart.csv')
IR >> head(3)

# Format date
minutes_BoW_sk2['oldDate'] =pd.to_datetime(minutes_BoW_sk2['file_name'],format='%Y%m%d',errors='ignore')
minutes_tfidf_sk['oldDate'] =pd.to_datetime(minutes_tfidf_sk['file_name'],format='%Y%m%d',errors='ignore')
IR['Date'] = pd.to_datetime(IR['date'],format='%Y/%m/%d',errors='ignore')

# It usually takes 20-22 days from the FOMC meeting to the publicly release of FOMC minutes
minutes_BoW_sk2['Date'] = minutes_BoW_sk2['oldDate'] + datetime.timedelta(days=23)
minutes_tfidf_sk['Date'] = minutes_tfidf_sk['oldDate'] + datetime.timedelta(days=23)

# Merge the date
bow_IR = pd.merge(IR,minutes_BoW_sk2,on = 'Date',how = 'left')
tfIdf_IR = pd.merge(IR,minutes_tfidf_sk,on = 'Date',how = 'left')
bow_IR_diff = bow_IR.dropna()
bow_IR_diff['rateChange'] = bow_IR_diff['fedRate'].shift(-1) - bow_IR_diff['fedRate']
tfIdf_IR_diff = tfIdf_IR.dropna()
tfIdf_IR_diff['rateChange'] = tfIdf_IR_diff['fedRate'].shift(-1) - tfIdf_IR_diff['fedRate']
```

### Logistic regression with FED rate moving direction

We first try to use OLS regression to predict the actual change of FED rate or use LSA to find out some interesting pattern in the documents, but none of them work well. 

So we turn our attention to predict the FED rate moving direction (up or down) by machine learning algorithm, using words that are statistically significant in logistic regression.

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

![best model results in different methods](https://github.com/Cometecomete/FOMC-minutes-analysis-and-interest-rate-prediction/blob/master/best%20model%20results%20in%20different%20methods.png)

Then we turn to the real estate industry level data, the predictive power is even better as below (you can check all the code in our code file):

![best model results for real estate industry](https://github.com/Cometecomete/FOMC-minutes-analysis-and-interest-rate-prediction/blob/master/best%20model%20results%20for%20real%20estate%20industry.png)

### Conclusion and others

We have done something else such as calculating the most correlated words in terms of FED rate change and get some meaningful words that have strong tendency, supporting our results to some extent. We think one possible explanation that our industry models work better is because that all the large companies in real estate industry pay a lot attention on the FOMC meetings and thus the FOMC minutes mean a lot to them, leading to a better correlation and prediction of the future performance in the industry.

We believe there still are so many improvements that could be done such as change the definition of the interest rate change and try to predict the moving direction after one month or six months after every minutes released; or use other industry data such as finance industry to see the predictive power; or simply improve the documents accuracy by manually checking.

But overall, we can conclude that using FOMC minutes we can have a relatively actual predictive power on FED interest rate change direction and relative industry performance.
