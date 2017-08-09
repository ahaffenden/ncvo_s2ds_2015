#!/usr/bin/python
"""
Created on Aug 20

@author: Anna MK
"""

# Running selected classifiers with hardcoded 3 of them: LogisticRegression(), RandomForestClassifier(), GaussianNB()

from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


import numpy as np
import pickle 
import os
import argparse
import ipdb

from ncvo_s2ds_2015.classifiers import ensemble_classifier 



def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  INPUT_DATA = os.path.join("..", "..", "data", "features", "bag_of_words_sparse_matrix.p")
  #INPUT_LABELS = os.path.join("..", "..", "data", "features", "hier_labels.pkl")

  parser.add_argument('--id', 
                      '--input-data', 
                      type=str, 
                      dest='input_data',
                      default=INPUT_DATA, 
                      help="File with input data in matrix format. Defaults to '%(default)s'")

#  parser.add_argument('--il', 
#                      '--input-labels',
#                      type=str, 
#                      dest='input_labels',
#                      default=INPUT_LABELS, 
#                      help="File with input labels. Defaults to '%(default)s'")

  ####PCA command line argument should go here (wether to do it or not and how much of the energy to be kept)


  args = parser.parse_args()

  np.random.seed(123)
  ipdb.set_trace()
  clf1 = LogisticRegression()
  clf2 = RandomForestClassifier()
  clf3 = GaussianNB()
  clf4 = DecisionTreeClassifier()
  clf5 = AdaBoostClassifier()

  print('5-fold cross validation:\n')

  #X = counted # sparse matrix input
  #X = tfidf
  #X = counted_bigr
  #y = data_str.iloc[:, 0]
  sparse_mat = pickle.load( open( args.input_data, "rb" ) )

  X = sparse_mat.iloc[:, 2]

  y = sparse_mat.iloc[:, 1]

  ###### if command line argument for PCA is True, then perform PCA on X here!

  # Ensemble classifier
  eclf = ensemble_classifier.EnsembleClassifier(clfs=[clf1, clf2, clf3, clf4, clf5], voting='hard')
  #eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[2,1,5]) # average probabilities, soft voting

  for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Decision Tree', 'AdaBoost', 'Ensemble']):

    scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    
    #scores = cross_validation.cross_val_score(clf3, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.5f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), label))

