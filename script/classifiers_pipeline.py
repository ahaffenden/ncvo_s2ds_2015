#!/usr/bin/python
"""
Created on Aug 20

@author: Anna MK
"""

# using split classifiers- different for subsets of features 

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from sklearn import datasets
import pickle 
import math

from ncvo_s2ds_2015.classifiers import ensemble_classifier 

# toy data set
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()

#X = pickle.load(open('../../data/features/hier_data.pkl', 'rb'))
#X = X.toarray()
#y = pickle.load(open('../../data/features/hier_labels.pkl', 'rb'))


N1 = math.floor(X.shape[1]/2)
Nend = X.shape[1]
vec1 = range(0, N1)
vec2 = range(N1, Nend)

pipe1 = Pipeline([
               ('sel', ensemble_classifier.ColumnSelector(vec1)),    # use only the 1st feature
               ('clf', clf1)])

pipe2 = Pipeline([
               ('sel', ensemble_classifier.ColumnSelector(vec2)), # use the 1st and 2nd feature
               ('dim', LDA(n_components=1)),    # Dimensionality reduction via LDA
               ('clf', clf2)])

eclf =  ensemble_classifier.EnsembleClassifier([pipe1, pipe2])
scores = cross_validation.cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), 'pipeline classifier'))

