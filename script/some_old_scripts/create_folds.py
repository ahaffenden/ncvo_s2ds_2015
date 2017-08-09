#! /usr/bin/env python
# Tue Aug 11 08:57:52 BST 2015
# Ivana Chingovska

"""
This script creates fold indices for cross-fold validation for the items in a particular file.
"""

import os
import argparse
import pandas
#import ipdb
import numpy
from ncvo_s2ds_2015.helpers import helpers


def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    
  # this is the default input directory if nothing is passed
  INPUT_FILE = os.path.join("..", "..", "data", "iTrain.csv") 
  OUTPUT_FILE = os.path.join("..", "..", "data", "folds", "iTrain.csv") 

  parser.add_argument('-i', 
                      '--input-file', 
                      type=str, 
                      dest='input_file',
                      default=INPUT_FILE, 
                      help="File to be processed. Defaults to '%(default)s'")

  parser.add_argument('--ns', 
                      '--num-samples', 
                      dest='num_samples', 
                      default=None,
                      type=int, 
                      help='The number of samples')

  parser.add_argument('--nf', 
                      '--num-folds', 
                      dest='num_folds', 
                      default=5,
                      type=int, 
                      help='The number of folds')
     
  parser.add_argument('-o', 
                      '--output-file', 
                      type=str, 
                      dest='output_file', 
                      default=OUTPUT_FILE, 
                      help="File to be used to save the fold indices. Defaults to '%(default)s'")

    
  args = parser.parse_args()
      
  # read the data    
  if args.num_samples == None:
    data_in = pandas.read_csv(args.input_file, encoding="ISO-8859-1")
    num_samples = len(data_in)
  else:
    num_samples = args.num_samples  

  # generate the folds
  from random import shuffle
  indices_list = list(range(0, num_samples))
  shuffle(indices_list)
  retval = []
  folds_indices = [x % args.num_folds + 1 for x in indices_list]

  # write the indices to file
  helpers.ensure_dir(os.path.dirname(args.output_file))
  folds_indices_towrite = pandas.Series(numpy.array(folds_indices))
  folds_indices_towrite.to_csv(args.output_file, index=False) # write to file, but don't give row names

  
if __name__ == "__main__":
  main()
