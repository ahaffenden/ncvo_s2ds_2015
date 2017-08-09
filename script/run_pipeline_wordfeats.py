#!/usr/bin/python
# Austin Haffenden
#Tue Aug 25 11:49:35 BST 2015

# Running a word tokens extraction and classification pipeline

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
import pandas

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
  INPUT_FILE = os.path.join("..", "..", "data", "features", "bow_string_input_dframe.p")
  OUTPUT_FILE_DESC = os.path.join("..", "..", "data", "output", 'predicted_labels_ensemble.csv')

  parser.add_argument('-i', 
                      '--input-file', 
                      type=str, 
                      dest='input_file',
                      default=INPUT_FILE, 
                      help="File with the list of item classes and features. Defaults to '%(default)s'")

  parser.add_argument('-g', 
                      '--gridres-file', 
                      type=str, 
                      dest='gridres_file',
                      default=None, 
                      help="File with the best parameters of the grid search. Defaults to '%(default)s'")

  parser.add_argument('-n',
                      '-ngrams',
                      dest = 'nGrams',
                      type = int,
                      default = 1, 
                      nargs='+',
                      help = 'Defines how to split words by ngrams. Default is tokenized to one word ngrams'
                      )

  parser.add_argument('--ti', 
                      dest='tf_idf', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, TfIdf features will be used')

  parser.add_argument('-c', 
                      '--classifier', 
                      type=str, 
                      dest='classifier',
                      default="ensemble", 
                      help="The classifier to be used. Defaults to '%(default)s'")

  parser.add_argument('--cat',
                      type=str, 
                      dest='category', 
                      default='income-type',
                      choices=('income-type','income-source','expenditure-type'),
                      help="The type of categorization. Defaults to '%(default)s'")

  parser.add_argument('--od', 
                      '--output-file-desc', 
                      type=str, 
                      dest='output_file_desc', 
                      default=OUTPUT_FILE_DESC, 
                      help="A csv file to output the predicted labels. Defaults to '%(default)s'")
 

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
  
  if args.gridres_file is not None:
    gridres_params = pickle.load(open(args.gridres_file, "rb" ))[0]
  else:
    gridres_params = None

  if args.classifier == 'decision-tree':
    max_depth = gridres_params['clf__max_depth'] if gridres_params is not None else 100
    clf = tree.DecisionTreeClassifier(max_depth=max_depth, criterion='entropy')
    print("Will run a decision tree with max depth=%d" % max_depth) 
  elif args.classifier == 'random-forest':
    max_depth = gridres_params['clf__max_depth'] if gridres_params is not None else 100
    clf = ensemble.RandomForestClassifier(max_depth=max_depth, criterion='entropy')
    print("Will run a random forest with max depth=%d" % max_depth) 
  elif args.classifier == 'logistic-regression':
    C = gridres_params['clf__C'] if gridres_params is not None else 1
    clf = linear_model.LogisticRegression(C=C)
    print("Will run logistic regressor with C=%d" % C)
  else: # ensemble
    clf1 = tree.DecisionTreeClassifier(max_depth=100, criterion='entropy')
    clf2 = ensemble.RandomForestClassifier()
    clf3 = linear_model.LogisticRegression()
    clf = ensemble_classifier.EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard')    

  vectorizer = feature_extraction.text.CountVectorizer( analyzer='word', #whether should be made ofword or char n-grams
                 binary=False, # if True all non-zero counts are set to one - used for probabilistic mapping
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

  labels_predicted = [-1] * len(labels_orig) 

  accuracy = []

  for train_idx, dev_idx in k_fold: 
    data_train = [data_orig[i] for i in train_idx]
    data_dev =  [data_orig[i] for i in dev_idx]
    labels_train = [labels_orig[i] for i in train_idx]
    labels_dev =  [labels_orig[i] for i in dev_idx]
  
    ppl.fit(data_train, labels_train)
    predicted_dev = ppl.predict(data_dev)
    labels_predicted = set_all_predicted(predicted_dev, labels_predicted, dev_idx)

    accuracy += [metrics.accuracy_score(labels_dev, predicted_dev)]
  
  print("Accuracy of the %s classifier: %.4f +- %.4f" % (args.classifier, numpy.mean(accuracy), numpy.std(accuracy)))

  # Save the predicted classes  
  to_dump = [labels_orig, labels_predicted]
  helpers.ensure_dir(os.path.dirname(args.output_file_desc))

  #create a dataframe to output type class, predicted type class and description data
  if args.category == 'income-type' or args.category == 'expenditure-type':
    dump_op_desc = pandas.DataFrame({'frID': frID,
                                    'type_class': labels_orig, 
                                   'type_class_predicted': labels_predicted,
                                   'description': data_orig})
  else:
    dump_op_desc = pandas.DataFrame({'frID': frID, 
                                    'source_class': labels_orig, 
                                   'source_class_predicted': labels_predicted,
                                   'description': data_orig})
  dump_op_desc.to_csv(args.output_file_desc)
  

if __name__ == "__main__":
  main()

  

  
