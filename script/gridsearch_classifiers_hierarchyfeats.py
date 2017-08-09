#!/usr/bin/python
#Ivana Chingovska
#Mon Aug 24 15:28:57 BST 2015

# Running a hierarchical tokens extraction and classification pipeline

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import ensemble
from sklearn import pipeline
from sklearn import tree
from sklearn import feature_extraction
from sklearn import metrics
from sklearn import grid_search
import numpy
import pickle 
import os
import argparse

from ncvo_s2ds_2015.classifiers import ensemble_classifier
from ncvo_s2ds_2015.helpers import helpers 



def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  INPUT_FILE = os.path.join("..", "..", "data", "features", "hierarchy-word-features.pkl")
  OUTPUT_FILE = os.path.join("..", "..", "data", "params", 'params-decisiontree.pkl')

  parser.add_argument('-i', 
                      '--input-file', 
                      type=str, 
                      dest='input_file',
                      default=INPUT_FILE, 
                      help="File with the list of item classes and features. Defaults to '%(default)s'")

  parser.add_argument('-c', 
                      '--classifier', 
                      type=str, 
                      dest='classifier',
                      default="decision-tree", 
                      help="The classifier to be used. Defaults to '%(default)s'")

  parser.add_argument('-o', 
                      '-output-file', 
                      type=str, 
                      dest='output_file', 
                      default=OUTPUT_FILE, 
                      help="A file to output the predicted labels. Defaults to '%(default)s'")

  

  args = parser.parse_args()

  # Read the input dictionary
  type_classes, source_classes, token_container = pickle.load(open(args.input_file, "rb"))
  type_dict = {'IGI': 0, 'IC': 1, 'IV': 2, 'IG': 3}
  # get all the label data
  labels_orig = [type_dict[x] for x in type_classes] 
  data_orig = token_container

  if args.classifier == 'decision-tree':
    import ipdb; ipdb.set_trace()
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    parameters = {
    'clf__max_depth': [i for i in range(25, 200, 25)],
  } 
  elif args.classifier == 'random-forest':
    clf = ensemble.RandomForestClassifier(criterion='entropy')
    parameters = {
    'clf__max_depth': [i for i in range(25, 200, 25)],
  }
  else: #'logistic-regression':
    clf = linear_model.LogisticRegression()
    parameters = {
    'clf__C': [0.5, 1, 5, 10],
  }

  ppl = pipeline.Pipeline([
    ('vectorizer', feature_extraction.DictVectorizer(sparse=True)), #sparse=True
    ('clf', clf),
  ])
  

  gs = grid_search.GridSearchCV(ppl, parameters, verbose=1, cv=5)
  gs.fit(data_orig, labels_orig)

  print(gs.best_params_, gs.best_score_)

  helpers.ensure_dir(os.path.dirname(args.output_file))
  pickle.dump([gs.best_params_, gs.best_params_], open(args.output_file, "wb" ))
  

  
