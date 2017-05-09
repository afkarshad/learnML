# -*- coding: utf-8 -*-
"""
Created on Tue May  9 22:19:16 2017

@author: Afkar
"""

import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas as pnd
import sklearn
import _pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
#from sklearn.learning_curve import learning_curve

def split_into_tokens(text):
    text = str(text)
    return TextBlob(text).words

def split_into_lemmas(text):
    text = str(text).lower()
    words = TextBlob(text).words
    return [word.lemma for word in words]

data = pnd.read_csv('../Datasets/YouTube-Spam-Collection-v1/Youtube01-Psy.csv')
headers = list(data)
description = data.groupby('CLASS').describe()
data['LENGTH'] = data['CONTENT'].map(lambda text: len(text))
#print (data.head())
#data.LENGTH.plot(bins=20, kind='hist')
#print(data.LENGTH.describe())
#data.hist(column='LENGTH', by='CLASS', bins=50)
#data_lemma = data.CONTENT.head().apply(split_into_lemmas)

tokens = CountVectorizer(analyzer=split_into_lemmas).fit(data.CONTENT)
#print(len(tokens.vocabulary_))

data_tokenized = tokens.transform(data.CONTENT)
#print(data_tokenized.shape)

tfidf_transformer = TfidfTransformer().fit(data_tokenized)

#pembobotan / weighting
data_tfidf = tfidf_transformer.transform(data_tokenized)
#print(data_tfidf.shape)

spam_detector = MultinomialNB().fit(data_tfidf, data['CLASS'])
index = 7
test_row = tfidf_transformer.transform(data_tokenized[index])
#print(test_row)
#print ('predicted:', spam_detector.predict(test_row)[0])
#print ('expected:', data.CLASS[index])

predictions = spam_detector.predict(data_tfidf)

print ('accuracy:', accuracy_score(data.CLASS, predictions))
print ('confusion matrix:\n', confusion_matrix(data.CLASS, predictions))
print (classification_report(data.CLASS, predictions))