#! /usr/bin/env python
# Ivana Chingovska
#Mon Aug 17 16:58:14 BST 2015

"""
This script selects the most useful tokens (hierarchical level phrases, as well as words) and saves them in a file.
"""

import os
import argparse
import pandas
import ipdb
import numpy
import scipy.sparse
import string
import nltk

from ncvo_s2ds_2015.helpers import helpers

def find_most_common_tokens(data, num_tokens):
    all_tokens = []
    for row in data.iterrows():
      all_tokens += (list(row[1])) # add all elements to a list row by row
      
    all_tokens = [x for x in all_tokens if str(x) != 'nan']

    all_tokens = [x.strip() for x in all_tokens] #remove spaces
    all_tokens = [x.strip(string.punctuation) for x in all_tokens] # remove punctuation

    token_occurence = nltk.FreqDist(all_tokens)
    most_common = token_occurence.most_common(num_tokens) # finding the most common tokens
    
    return [x[0] for x in most_common]

def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    
  # this is the default input directory if nothing is passed
  INPUT_HT = os.path.join("..", "..", "data", "features", "data_frame_hierarchy.csv")
  INPUT_WT = os.path.join("..", "..", "data", "features", "data_frame_words.csv")
  OUTPUT_DIR = os.path.join("..", "..", "data", "features") 


  parser.add_argument('--ht', 
                      '--hierarchical-tokens-file', 
                      type=str, 
                      dest='input_ht', 
                      default=INPUT_HT, 
                      help="A dataframe file containing the hierarchical level tokens for each item in the dataset. Defaults to '%(default)s'")

  parser.add_argument('--wt', 
                      '--word-tokens-file', 
                      type=str, 
                      dest='input_wt', 
                      default=INPUT_WT, 
                      help="A dataframe file containing the word tokens for each item in the dataset. Defaults to '%(default)s'")

  parser.add_argument('--nh', 
                      '--num-hierarchy-tokens', 
                      dest='num_hierarchy_tokens', 
                      default=20,
                      type=int, 
                      help="The number of the most frequent hierarchical tokens to use per class. Defaults to '%(default)s'")

  parser.add_argument('--nw', 
                      '--num-word-tokens', 
                      dest='num_word_tokens', 
                      default=20,
                      type=int, 
                      help="The number of the most frequent hierarchical tokens to use per class. Defaults to '%(default)s'")

  parser.add_argument('-o', 
                      '--output-dir', 
                      type=str, 
                      dest='output_dir', 
                      default=OUTPUT_DIR, 
                      help="Directory to save the selected tokens. Defaults to '%(default)s'") 

  args = parser.parse_args()
      
  data_hierarchy  = pandas.read_csv(args.input_ht, encoding="ISO-8859-1")
  data_words  = pandas.read_csv(args.input_wt, encoding="ISO-8859-1")

  # Remove samples from classes which are noisy and not useful ("I", "IO")
  data_hierarchy = data_hierarchy[data_hierarchy.type_class != 'I']
  data_hierarchy = data_hierarchy[data_hierarchy.type_class != 'IO']
  data_words = data_words[data_words.type_class != 'I']
  data_words = data_words[data_words.type_class != 'IO']

  # Read the available classes and labels from the data
  labels_hierarchy = data_hierarchy.type_class.unique()
  labels_words = data_words.type_class.unique()
  columns_hierarchy = data_hierarchy.columns
  columns_words = data_hierarchy.columns

  hierarchy_tokens = []
  # Finding the N most frequent hierarchy tokens
  print("Processing hierarchical tokens...")
  for label in labels_hierarchy:
    print("Processing %s class..." % label)
    data_by_label = data_hierarchy[data_hierarchy.type_class == label] # filter only the items with the particular label
    descr = data_by_label[columns_hierarchy[2:]] # take only the columns that represent description
    hierarchy_tokens += find_most_common_tokens(descr, args.num_hierarchy_tokens)

  word_tokens = []
  # Finding the N most frequent word tokens
  print("Processing word tokens...")
  for label in labels_words:
    print("Processing %s class..." % label)
    data_by_label = data_words[data_words.type_class == label] # filter only the items with the particular label
    descr = data_by_label[columns_words[2:]] # take only the columns that represent description
    word_tokens += find_most_common_tokens(descr, args.num_word_tokens)  

  #Save the tokens
  import ipdb; ipdb.set_trace()
  helpers.ensure_dir(args.output_dir)
  hierarchy_tokens_towrite = pandas.Series(pandas.Series(numpy.array(hierarchy_tokens)).unique()) # take each token just once
  hierarchy_tokens_towrite.to_csv(os.path.join(args.output_dir, 'hierarchy_tokens.csv'), index=False) # write to file, but don't give row names
  word_tokens_towrite = pandas.Series(pandas.Series(numpy.array(word_tokens)).unique())
  word_tokens_towrite.to_csv(os.path.join(args.output_dir, 'word_tokens.csv'), index=False) # write to file, but don't give row names

if __name__ == "__main__":
  main()
