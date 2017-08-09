#! /usr/bin/env python
# Ivana Chingovska
#Tue Aug 11 15:45:52 BST 2015

"""
This script trains a decision tree using certain folds
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
from sklearn import feature_extraction
from sklearn import metrics

from ncvo_s2ds_2015.helpers import helpers
from ncvo_s2ds_2015.helpers import data_manipulation

def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    
  # this is the default input directory if nothing is passed
  INPUT_FILE = os.path.join("..", "..", "data", "features", "hierarchy-word-features.pkl")
  OUTPUT_DIR = os.path.join("..", "..", "data", "classifiers", "decision_trees")
  FOLD_FILE = os.path.join("..", "..", "data", "folds", "newfolds.csv") 

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
                      help="The fold number to be EXCLUDED when creating the master set (If there are N folds, N-1 folds are used for creating the master set). If 0, all the items will be considered. If None, will train the classifier for all the folds in a loop. Defaults to '%(default)s'")

  parser.add_argument('--ff', 
                      '--fold-file', 
                      type=str, 
                      dest='foldFile', 
                      default=FOLD_FILE, 
                      help="Fold file containing the cross-fold validation indices. Defaults to '%(default)s'")
   
  parser.add_argument('-o', 
                      '--output-dir', 
                      type=str, 
                      dest='output_dir', 
                      default=OUTPUT_DIR, 
                      help="Directory to save the decision trees. Defaults to '%(default)s'")


  args = parser.parse_args()

  # Read the input dictionary
  type_classes, source_classes, token_container = pickle.load(open(args.input_file, "rb"))
  type_dict = {'IGI': 0, 'IC': 1, 'IV': 2, 'IG': 3}
  # get all the label data
  labels_orig = [type_dict[x] for x in type_classes]    
  
  # Create output directory
  helpers.ensure_dir(args.output_dir)

  dtree = tree.DecisionTreeClassifier(random_state=0, max_depth=100, criterion='entropy')
  vectorizer = feature_extraction.DictVectorizer(sparse=True)

  if args.fold_number == None: # loop over all folds and create the classifier
    print("Training will be done iteratively for all folds...\n")
    cross_fold_indices = pandas.read_csv(args.foldFile, header=None)[0]
    for fold in cross_fold_indices.unique():
      print("Training classifier for fold %d...\n" % fold)
      
      data_fold = [token_container[i] for i in range(len(cross_fold_indices)) if cross_fold_indices[i] != fold] 
      labels_fold = [labels_orig[i] for i in range(len(cross_fold_indices)) if cross_fold_indices[i] != fold]
      data_matrix = vectorizer.fit_transform(data_fold)
      dtree.fit(data_matrix, labels_fold)
      joblib.dump(dtree, os.path.join(args.output_dir, 'tree-fold%d.pkl' % fold))
      joblib.dump(vectorizer, os.path.join(args.output_dir, 'vectorizer-fold%d.pkl' % fold))
      labels_pred = dtree.predict(data_matrix)
      score = metrics.accuracy_score(labels_fold, labels_pred)
      print("Accuracy on fold %d (train set): %.5f" % (fold, score))
      tree.export_graphviz(dtree, out_file= os.path.join(args.output_dir, 'tree-fold%d.dot' % fold), max_depth=5)#, feature_names = master_in.values)

  elif args.fold_number == 0: # use all the data to train the classifier
    print("Training classifier for the full set...\n")
    data_matrix = vectorizer.fit_transform(token_container)
    import ipdb; ipdb.set_trace()
    dtree.fit(data_matrix, labels_orig)
    joblib.dump(dtree, os.path.join(args.output_dir, 'tree.pkl')) 
    joblib.dump(vectorizer, os.path.join(args.output_dir, 'vectorizer.pkl')) 
    tree.export_graphviz(dtree, out_file= os.path.join(args.output_dir, 'tree.dot'), max_depth=5)#, feature_names = master_in.values)

  else: # create classifier for a particular fold
    fold = args.fold_number
    print("Training classifier for fold %d...\n" % fold)
    cross_fold_indices = pandas.read_csv(args.foldFile, header=None)[0]
    data_fold = [token_container[i] for i in range(len(cross_fold_indices)) if cross_fold_indices[i] != fold] 
    labels_fold = [labels_orig[i] for i in range(len(cross_fold_indices)) if cross_fold_indices[i] != fold]
    data_matrix = vectorizer.fit_transform(data_fold)
    dtree.fit(data_matrix, labels_fold)
    joblib.dump(dtree, os.path.join(args.output_dir, 'tree-fold%d.pkl' % args.fold_number)) 
    joblib.dump(vectorizer, os.path.join(args.output_dir, 'vectorizer-fold%d.pkl' % args.fold_number)) 
    labels_pred = dtree.predict(data_matrix)
    score = metrics.accuracy_score(labels_fold, labels_pred)
    print("Accuracy on fold %d (train set): %.5f" % (fold, score))
  
if __name__ == "__main__":
  main()
