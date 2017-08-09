#!/usr/bin/python
#Ivana Chingovska
#Mon Aug 24 11:25:38 BST 2015

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
import numpy
import pickle 
import os
import argparse

from ncvo_s2ds_2015.classifiers import ensemble_classifier
from ncvo_s2ds_2015.helpers import helpers 

def set_all_predicted(predicted, all_predicted, ind):
  for i in range(len(ind)):
    all_predicted[ind[i]] = predicted[i]
  return all_predicted

def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  INPUT_FILE = os.path.join("..", "..", "data", "features", "hierarchy-word-features.pkl")
  OUTPUT_FILE = os.path.join("..", "..", "data", "output", 'predicted_labels_ensemble.pkl')

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
                      default="ensemble", 
                      help="The classifier to be used. Defaults to '%(default)s'")

  parser.add_argument('--orig-labels', 
                      dest='orig_labels', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, the original data labels will be stored. Otherwise, they will be coded as integers')

  parser.add_argument('--cat',
                      type=str, 
                      dest='category', 
                      default='income-type',
                      choices=('income-type','income-source','expenditure-type'),
                      help="The type of categorization. Defaults to '%(default)s'")

  parser.add_argument('-o', 
                      '-output-file', 
                      type=str, 
                      dest='output_file', 
                      default=OUTPUT_FILE, 
                      help="A file to output the predicted labels. Defaults to '%(default)s'")

  

  args = parser.parse_args()

  # Read the input dictionary
  type_classes, source_classes, token_container = pickle.load(open(args.input_file, "rb"))
  
  import ipdb; ipdb.set_trace()
  # get all the label data
  if args.category == 'income-type' or args.category == 'expenditure-type':
    labels_orig = [str(i) for i in type_classes] # converting them to strings if they are not strings already
  else:
    labels_orig = [str(i) for i in source_classes] # converting them to strings if they are not strings already
  #labels_orig = [type_dict[x] for x in type_classes] 
  data_orig = token_container

  if args.classifier == 'decision-tree':
    clf = tree.DecisionTreeClassifier(max_depth=100, criterion='entropy')
  elif args.classifier == 'random-forest':
    clf = ensemble.RandomForestClassifier()
  elif args.classifier == 'logistic-regression':
    clf = linear_model.LogisticRegression()
  else: # ensemble
    clf1 = tree.DecisionTreeClassifier(max_depth=100, criterion='entropy')
    clf2 = ensemble.RandomForestClassifier()
    clf3 = linear_model.LogisticRegression()
    clf = ensemble_classifier.EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard')    

  ppl = pipeline.Pipeline([
    ('vectorizer', feature_extraction.DictVectorizer(sparse=True)), #sparse=True
    ('clf', clf),
  ])

  k_fold = cross_validation.StratifiedKFold(labels_orig, 5, shuffle=True)

  #labels_predicted = numpy.array([-1] * len(labels_orig), dtype='int')
  labels_predicted = [-1] * len(labels_orig) 

  accuracy = []

  for train_idx, dev_idx in k_fold: 
    data_train = [data_orig[i] for i in train_idx]
    data_dev =  [data_orig[i] for i in dev_idx]
    labels_train = [labels_orig[i] for i in train_idx]
    labels_dev =  [labels_orig[i] for i in dev_idx]
  
    ppl.fit(data_train, labels_train)
    predicted_dev = ppl.predict(data_dev)
    #labels_predicted[dev_idx] = predicted_dev
    labels_predicted = set_all_predicted(predicted_dev, labels_predicted, dev_idx)

    accuracy += [metrics.accuracy_score(labels_dev, predicted_dev)]
  
  print("Accuracy of the %s classifier: %.4f +- %.4f" % (args.classifier, numpy.mean(accuracy), numpy.std(accuracy)))

  # Save the predicted classes  
  #inv_type_dict = {v: k for k, v in type_dict.items()}
  
  to_dump = [labels_orig, labels_predicted]
  helpers.ensure_dir(os.path.dirname(args.output_file))
  pickle.dump(to_dump, open(args.output_file, "wb"))

  '''
  predicted_type_classes = [inv_type_dict[x] for x in labels_predicted]
  if args.orig_labels: # save the original labels
    predicted_type_classes = [inv_type_dict[x] for x in labels_predicted]
    to_dump = [type_classes, predicted_type_classes]
  else: # save the labels codified with integer numbers, as well as the decoding dictionary
    to_dump = [labels_orig, labels_predicted, inv_type_dict]
  '''
  



  

  