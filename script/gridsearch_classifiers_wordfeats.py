#!/usr/bin/python
#Ivana Chingovska
#Sun Aug 30 19:46:58 BST 2015

# Running a words tokens extraction and classification pipeline

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
from sklearn import pipeline
from sklearn import tree
from sklearn import feature_extraction
from sklearn import metrics
from sklearn import grid_search
import pandas
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

  parser.add_argument('-n',
                      '-ngrams',
                      dest = 'nGrams',
                      type = int,
                      default = 1, 
                      help = 'Defines how to split words by ngrams. Default is tokenized to one word ngrams'
                      )

  parser.add_argument('--ti', 
                      dest='tf_idf', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, TfIdf features will be used')

  parser.add_argument('-b', 
                      dest='binary', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, CountVecorizer will use binary counts instead or frequency counts')


  parser.add_argument('-c', 
                      '--classifier', 
                      type=str, 
                      dest='classifier',
                      default="decision-tree", 
                      help="The classifier to be used. Defaults to '%(default)s'")

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
  
  data_in = pandas.read_pickle(args.input_file)
  type_classes = list(data_in['type_class'])
  source_classes = list(data_in['source_class'])
  frID = list(data_in['frID'])
  data_orig = data_in['description']

  if args.category == 'income-type' or args.category == 'expenditure-type':
    labels_orig = [str(i) for i in type_classes] # converting them to strings if they are not strings already
  else:
    labels_orig = [str(i) for i in source_classes] # converting them to strings if they are not strings already

  if args.classifier == 'decision-tree':
    #import ipdb; ipdb.set_trace()
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    parameters = {
    'clf__max_depth': [i for i in range(25, 200, 25)], #200
  } 
  elif args.classifier == 'random-forest':
    clf = ensemble.RandomForestClassifier(criterion='entropy')
    parameters = {
    'clf__max_depth': [i for i in range(25, 200, 25)],
  }
  elif args.classifier == 'logistic-regression': #'logistic-regression':
    clf = linear_model.LogisticRegression()
    parameters = {
    'clf__C': [0.5, 1, 5, 10],
  }
  else: # SVM
    clf = svm.SVC()
    parameters = {
    'clf__C': [0.5, 1.0, 5.0, 10], 
    }

  vectorizer = feature_extraction.text.CountVectorizer( analyzer='word', #whether should be made ofword or char n-grams
                 binary=args.binary, # if True all non-zero counts are set to one - used for probabilistic mapping
                 decode_error= 'strict', # Instruction on what to do if a byte sequence is given to analyze that contains characters not of the given encoding
                 #dtype='numpy.int64', # Type of the matrix returned by fit_transform() or transform()
                 encoding="ISO-8859-15", # 
                 input='content', # can be 'file', 'filename' or 'content'
                 lowercase=False, #Convert all characters to lowercase before tokenizing. 
                 max_df=1.0, # When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None."
                 max_features=None, # If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. This parameter is ignored if vocabulary is not None.
                 ngram_range=(1, args.nGrams), # The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.
                 preprocessor=None, # Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.
                 stop_words=None, #     
                 min_df=1,
                 strip_accents=None, 
                 token_pattern = '(?u)\\b\\w\\w+\\b',
                 tokenizer=None, 
                 vocabulary=None )

  if args.tf_idf == True:
    transformer = feature_extraction.text.TfidfTransformer()
    ppl = pipeline.Pipeline([
      ('vectorizer', vectorizer),
      ('transformer', transformer),
      ('clf', clf),
    ])
  else:
    ppl = pipeline.Pipeline([
      ('vectorizer', vectorizer),
      ('clf', clf),
    ])
  

  k_fold = cross_validation.StratifiedKFold(labels_orig, 5, shuffle=True)
  gs = grid_search.GridSearchCV(ppl, parameters, verbose=3, cv=k_fold)
  gs.fit(data_orig, labels_orig)

  print(gs.best_params_, gs.best_score_)

  helpers.ensure_dir(os.path.dirname(args.output_file))
  pickle.dump([gs.best_params_, gs.best_score_], open(args.output_file, "wb" ))
  

  
