#! /usr/bin/env python
# Ivana Chingovska
# Mon Aug 17 16:51:04 BST 2015

"""
This script creates a sparse matrix containing the features. Each feature is 0 or 1 depending on whether the item contains the token in a particular hierarchical level. The tokens are both hierarchical level phrases and words. 
"""

import os
import argparse
import pandas
import ipdb
import numpy
import string
import scipy.sparse
import pickle

from ncvo_s2ds_2015.helpers import helpers

def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    
  # this is the default input directory if nothing is passed
  # this is the default input directory if nothing is passed
  INPUT_HS = os.path.join("..", "..", "data", "features", "data_frame_hierarchy.csv")
  INPUT_HT = os.path.join("..", "..", "data", "features", "hierarchical-tokens.csv")
  OUTPUT_FILE = os.path.join("..", "..", "data", "features", "hierarchy-features.pkl")


  parser.add_argument('--hs', 
                      '--hierarchical-split', 
                      type=str, 
                      dest='input_hs', 
                      default=INPUT_HS, 
                      help="A dataframe file containing the hierarchical level tokens for each item in the dataset. Defaults to '%(default)s'")

  parser.add_argument('--ht', 
                      '--hierarchical-tokens', 
                      type=str, 
                      dest='input_ht', 
                      default=INPUT_HT, 
                      help="A file containing the selected hierarchical level tokens. Defaults to '%(default)s'")

  parser.add_argument('-o', 
                      '-output-file', 
                      type=str, 
                      dest='output_file', 
                      default=OUTPUT_FILE, 
                      help="A file to output the dictionary. Defaults to '%(default)s'")


  args = parser.parse_args()
  
  # Read the data
  data_hierarchy  = pandas.read_csv(args.input_hs, encoding="ISO-8859-1")
  
  # Read the tokens
  hierarchy_tokens = pandas.read_csv(args.input_ht, header=None)[0]

  # Remove samples from classes which are noisy and not useful ("I", "IO")
  data_hierarchy = data_hierarchy[data_hierarchy.type_class != 'I']
  data_hierarchy = data_hierarchy[data_hierarchy.type_class != 'IO']

  columns_hierarchy = data_hierarchy.columns

  descr_hierarchy = data_hierarchy[columns_hierarchy[2:]]

  #import ipdb; ipdb.set_trace()
  hierarchy_token_container = []
  for row in descr_hierarchy.iterrows():
    row_dict = {} # initialize the 
    rowlist = list(row[1])
    rowlist = [x for x in rowlist if str(x) != 'nan']
    rowlist = [x.strip() for x in rowlist] #remove spaces
    rowlist = [x.strip(string.punctuation) for x in rowlist] # remove punctuation
    for token in hierarchy_tokens:
      try:
        ind = rowlist.index(token)
        if ind <= 3:
          row_dict[token] = rowlist.index(token)
        else:
          row_dict[token] = 4
      except ValueError:
        pass
    hierarchy_token_container += [row_dict]      
  
  # Extracting the sample classes into more convenient format
  type_class_list = list(data_hierarchy['type_class'])
  source_class_list = list(data_hierarchy['source_class'])

  # Save everything alltogether in a pkl file  
  to_dump = [type_class_list, source_class_list, hierarchy_token_container]
  helpers.ensure_dir(os.path.dirname(args.output_file))
  pickle.dump(to_dump, open(args.output_file, "wb" ))


  

if __name__ == "__main__":
  main()
