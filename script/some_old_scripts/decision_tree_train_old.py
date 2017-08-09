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
  INPUT_FILE = os.path.join("..", "..", "data", "features", "data_frame.csv") 
  OUTPUT_DIR = os.path.join("..", "..", "data", "classifiers", "decision_trees")
  FOLD_FILE = os.path.join("..", "..", "data", "folds", "iTrain.csv") 
  INPUT_MS = os.path.join("..", "..", "data", "features", "master_set.csv")

  parser.add_argument('-i', 
                      '--input-file', 
                      type=str, 
                      dest='input_file',
                      default=INPUT_FILE, 
                      help="File with features. The classes will be taken from this file. Defaults to '%(default)s'")

  parser.add_argument('--ms', 
                      '--master_set', 
                      dest='master_set', 
                      default=INPUT_MS, 
                      help="The input file containing the master set. Defaults to '%(default)s'")

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

  type_dict = {'IV':1, 'IG':2, 'IC':3, 'I':4, 'IGI':5, 'IO':6}
      
  # read the input matrix and the labels
  data_in = pandas.read_csv(args.input_file, encoding="ISO-8859-1")
  master_in = pandas.read_csv(args.master_set, encoding="ISO-8859-1", header=-1)

  labels_orig=data_in.type_class
  labels=pandas.Series([type_dict[x] for x in labels_orig])

  # Create output directory
  helpers.ensure_dir(args.output_dir)

  dtree = tree.DecisionTreeClassifier(random_state=0, max_depth=100, criterion='entropy')
  
  if args.fold_number == None: # loop over all folds and create the classifier
    print("Training will be done iteratively for all folds...\n")
    cross_fold_indices = pandas.read_csv(args.foldFile, header=None)[0]
    for fold in cross_fold_indices.unique():
      print("Training classifier for fold %d...\n" % fold)
      data_fold = data_in.loc[cross_fold_indices != fold,:]
      import ipdb; ipdb.set_trace()
      data_matrix = data_manipulation.binary_sparse_matrix(data_fold, master_in)
      labels_fold = labels[cross_fold_indices != fold]
      dtree.fit(data_matrix, labels_fold)
      joblib.dump(dtree, os.path.join(args.output_dir, 'tree-fold%d.pkl' % fold)) 
      tree.export_graphviz(dtree, out_file= os.path.join(args.output_dir, 'tree-fold%d.dot' % fold), max_depth=5, feature_names = master_in.values)

  elif args.fold_number == 0: # use all the data to train the classifier
    print("Training classifier for the full set...\n" % fold)
    data_matrix = data_manipulation.binary_sparse_matrix(data_in, master_in)
    dtree.fit(data_matrix, labels)
    joblib.dump(dtree, os.path.join(args.output_dir, 'tree.pkl')) 

  else: # create classifier for a particular fold
    print("Training classifier for fold %d...\n" % fold)
    cross_fold_indices = pandas.read_csv(args.foldFile, header=None)[0]
    data_fold = data_in.loc[cross_fold_indices != args.fold_number]
    data_matrix = data_manipulation.binary_sparse_matrix(data_fold, master_in)
    labels = labels[cross_fold_indices != args.fold_number]
    dtree.fit(data_matrix, labels)
    joblib.dump(dtree, os.path.join(args.output_dir, 'tree-fold%d.pkl' % args.fold_number)) 

  
if __name__ == "__main__":
  main()
