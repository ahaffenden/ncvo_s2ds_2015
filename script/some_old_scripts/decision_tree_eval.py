#! /usr/bin/env python
# Ivana Chingovska
#Wed Aug 12 17:59:33 BST 2015
"""
This script makes predictions about samples based on a decision tree classifier
"""

import os
import argparse
import pandas
import ipdb
import numpy
import pickle

from sklearn.externals import joblib
from scipy import sparse

from sklearn import tree
from sklearn import metrics

from ncvo_s2ds_2015.helpers import helpers
from ncvo_s2ds_2015.helpers import data_manipulation

def ensure_dir(directory):
  """Method that creates a directory if it does not exist"""
  if not os.path.exists(directory):
    os.makedirs(directory)

def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    
  # this is the default input directory if nothing is passed
  INPUT_FILE = os.path.join("..", "..", "data", "features", "hierarchy-word-features.pkl")
  CLASSIFIER_DIR = os.path.join("..", "..", "data", "classifiers", "decision_trees")
  FOLD_FILE = os.path.join("..", "..", "data", "folds", "newfolds.csv") 
  OUTPUT_DIR = os.path.join("..", "..", "data", "scores", "decision_trees")

  parser.add_argument('-i', 
                      '--input-file', 
                      type=str, 
                      dest='input_file',
                      default=INPUT_FILE, 
                      help="File with the list of item classes and features. Defaults to '%(default)s'")

  parser.add_argument('--fn', 
                      '--fold-number', 
                      dest='fold_number', 
                      default=None,
                      type=int, 
                      help="The fold number to be used for predicition. If 0, all the items will be considered. If None, will train the classifier for all the folds. Defaults to '%(default)s'")

  parser.add_argument('--ff', 
                      '--fold-file', 
                      type=str, 
                      dest='fold_file', 
                      default=FOLD_FILE, 
                      help="Fold file containing the cross-fold validation indices. Defaults to '%(default)s'")
   
  parser.add_argument('--cd', 
                      '--classifier-dir', 
                      type=str, 
                      dest='classifier_dir', 
                      default=CLASSIFIER_DIR, 
                      help="Directory where the trained decision trees are stored. Defaults to '%(default)s'")

  #parser.add_argument('--sm', 
  #                    '--sparse-matrix', 
  #                    dest='sparse_matrix', 
  #                    action='store_true', 
  #                    default=False, 
  #                    help='If set, the data will be transformed into a sparse matrix')

  parser.add_argument('-o', 
                      '--output-dir', 
                      type=str, 
                      dest='output_dir', 
                      default=OUTPUT_DIR, 
                      help="Directory to save the scores. Defaults to '%(default)s'")

    
  args = parser.parse_args()
      
  # Read the input dictionary
  type_classes, source_classes, token_container = pickle.load(open(args.input_file, "rb"))
  type_dict = {'IGI': 0, 'IC': 1, 'IV': 2, 'IG': 3}

  # get all the label data
  labels_orig = [type_dict[x] for x in type_classes]    
  
  # Create output directory
  helpers.ensure_dir(args.output_dir)
  
  # Read the fold indices
  cross_fold_indices = pandas.read_csv(args.fold_file, header=None)[0]

  if args.fold_number == None: # loop over all folds and create the classifier
    print("Evaluation will be done iteratively for all folds...\n")
    pred_labels = {}
    
    for fold in cross_fold_indices.unique():
      print("Evaluating fold %d...\n" % fold)
      # read the vectorizer and the decision tree
      vectorizer = joblib.load(os.path.join(args.classifier_dir, 'vectorizer-fold%d.pkl' % (fold)))
      dtree = joblib.load(os.path.join(args.classifier_dir, 'tree-fold%d.pkl' % (fold)))

      # read and transformthe data
      data_fold = [token_container[i] for i in range(len(cross_fold_indices)) if cross_fold_indices[i] == fold]
      data_matrix = vectorizer.transform(data_fold)

      pred_labels[fold] = dtree.predict(data_matrix)
      
  elif args.fold_number == 0: # use all the data to train the classifier
    print("Evaluation will be done on the full set...\n" % fold)
    # read the vectorizer and decision tree
    vectorizer = joblib.load(os.path.join(args.classifier_dir, 'vectorizer.pkl'))
    dtree = joblib.load(os.path.join(args.classifier_dir, 'tree.pkl'))

    #read the data
    data_matrix = vectorizer.transform(token_container)
  
    pred_labels = dtree.predict(data_matrix)

  else: # create classifier for a particular fold
    import ipdb; ipdb.set_trace()
    fold = args.fold_number
    print("Evaluation for fold %d...\n" % fold)
    
    # read the vectorizer and the decision tree
    vectorizer = joblib.load(os.path.join(args.classifier_dir, 'vectorizer-fold%d.pkl' % (fold)))
    dtree = joblib.load(os.path.join(args.classifier_dir, 'tree-fold%d.pkl' % (fold)))

    # read and transformthe data
    data_fold = [token_container[i] for i in range(len(cross_fold_indices)) if cross_fold_indices[i] == fold]
    data_matrix = vectorizer.transform(data_fold)

    pred_labels = dtree.predict(data_matrix)


  # do the evaluation
  if args.fold_number == None: # we iterate over all the folds
    for fold in cross_fold_indices.unique():
      labels_fold = [labels_orig[i] for i in range(len(cross_fold_indices)) if cross_fold_indices[i] == fold]
      pred_labels_fold = pred_labels[fold]
      score = metrics.accuracy_score(labels_fold, pred_labels_fold)
      print("Accuracy on fold %d: %.5f" % (fold, score))
  elif args.fold_number == 0:
    pass
  else:
    fold = args.fold_number
    cross_fold_indices = pandas.read_csv(args.fold_file, header=None)[0]
    labels_fold = [labels_orig[i] for i in range(len(cross_fold_indices)) if cross_fold_indices[i] == fold]
    score = metrics.accuracy_score(labels_fold, pred_labels)
    print("Accuracy on fold %d: %.5f" % (fold, score))




  print("Done!\n")
  
if __name__ == "__main__":
  main()
