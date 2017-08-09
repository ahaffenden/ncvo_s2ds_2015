#! /usr/bin/env python
"""
Created on Wed Aug 19 12:12:29 2015
Ivana Chingovska
Tue Aug 25 18:47:33 BST 2015
"""

import os
#import inspect
import ipdb
#import nltk
#import numpy
import argparse
import pandas
import pickle
import matplotlib.pyplot as plt 

from ncvo_s2ds_2015.helpers import helpers

def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    
  # this is the default input directory if nothing is passed
  # this default file still contains I and IO values so needs updating
  INPUT_ORIG_FILE = os.path.join("..", "..", "data", "output", 'orig_file.csv')
  INPUT_TYPE_FILE = os.path.join("..", "..", "data", "output", 'predicted_labels_ensemble_type.csv')
  INPUT_SOURCE_FILE = os.path.join("..", "..", "data", "output", 'predicted_labels_ensemble_source.csv')
  OUTPUT_FILENAME = os.path.join("..", "..", "data", "output", 'predicted_labels_ensemble_joined.csv')

  parser.add_argument('--io', 
                      '--input-orig', 
                      type=str, 
                      dest='input_orig', 
                      default=INPUT_ORIG_FILE, 
                      help='Dataframe with the original input samples')

  parser.add_argument('--it', 
                      '--input-type', 
                      type=str, 
                      dest='input_type', 
                      default=None,
                      help='Dataframe for the output of the type prediction')
 
  parser.add_argument('--is', 
                      '--input-source', 
                      type=str, 
                      dest='input_source', 
                      default=None, 
                      help='Dataframe for the output of the source prediction (if applicable)') 
  
  parser.add_argument('--jm',
                      '--just_missed',
                      dest='just_missed',
                      action='store_true', 
                      default=False, 
                      help='If set, only the missclassified samples will be written to a file') 

  parser.add_argument('-o', 
                      '-output-file', 
                      type=str, 
                      dest='output_file', 
                      default=OUTPUT_FILENAME, 
                      help="Filename of output file")                      
                      
  args = parser.parse_args()
  
  # Read original and classified data
  data_orig = pandas.read_csv(args.input_orig, encoding="ISO-8859-15", dtype='str' ) # read oiginal dataframe
  if args.input_type is not None:
    data_type = pandas.read_csv(args.input_type, encoding="ISO-8859-15", dtype='str' ) # read type predicitions
  if args.input_source is not None: #read source predictions
    data_source = pandas.read_csv(args.input_source, encoding="ISO-8859-15", dtype='str' )
  
  # drop columns which have the same name in original dataframe and the prediction dataframes
  if args.input_type is not None:
    data_type.drop(['description', 'type_class'], axis=1, inplace=True) 
  if args.input_source is not None:
    data_source.drop(['description', 'source_class'], axis=1, inplace=True)

  # join the data
  if args.input_type is not None and args.input_source is not None:
    result_df = pandas.merge(data_orig, data_type, on='frID', suffixes = ('_orig', '_pred'), how='left') # join original and type prediction dataframe
    result_df = pandas.merge(result_df, data_source, on='frID', suffixes = ('_orig', '_pred'), how='left') # join original and type prediction dataframe
  elif args.input_type is not None:  
    result_df = pandas.merge(data_orig, data_type, on='frID', suffixes = ('_orig', '_pred'), how='left') # join original and source prediction dataframe
  else:
    result_df = pandas.merge(data_orig, data_source, on='frID', suffixes = ('_orig', '_pred'), how='left') # join original and source prediction dataframe  
      
  # Rearrange the columns
  if args.input_type is not None and args.input_source is not None:
    result_df = result_df[['frID', 'description', 'type_class', 'type_class_predicted', 'source_class', 'source_class_predicted', 'ICNPO_category', 'nicename']] #rearrange the columns
  elif args.input_type is not None:
    result_df = result_df[['frID', 'description', 'type_class', 'type_class_predicted', 'ICNPO_category', 'nicename']] #rearrange the columns
  else:  
    result_df = result_df[['frID', 'description', 'source_class', 'source_class_predicted', 'ICNPO_category', 'nicename']] #rearrange the columns

  # Compute joined accuracy if both type_class and source_class are given
  if args.input_type is not None and args.input_source is not None:
    orig_type = result_df.type_class
    pred_type = result_df.type_class_predicted
    orig_source = result_df.source_class
    pred_source = result_df.source_class_predicted 
    
    # check number of correctly predicted labels for type and source
    type_common = orig_type == pred_type
    source_common = orig_source == pred_source  
    common_classifications = type_common & source_common

    common_accuracy = sum(common_classifications) / float(len(common_classifications))
    print("Joint accuracy: %.5f" % common_accuracy)

  # Save just the incorrectly classified samples 
  if args.just_missed == True:  
    if args.input_type is not None and args.input_source is not None:
      misclassified = result_df.type_class != result_df.type_class_predicted | result_df.source_class != result_df.source_class_predicted
    elif args.input_type is not None:
      misclassified = result_df.type_class != result_df.type_class_predicted
    else:
      misclassified = result_df.source_class != result_df.source_class_predicted 
  result_df = result_df[misclassified]    

  result_df.to_csv(args.output_file) # save the new dataframe

if __name__ == "__main__":
  main()

