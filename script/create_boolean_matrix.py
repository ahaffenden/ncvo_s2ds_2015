#! /usr/bin/env python
#Created on Fri Aug  7 17:10:28 2015
#@author: graeme
"""
This script creates the boolean matrix related to the master set for input to the 
decision tree classifier.Tthe type and source (if income) as the 
first and second column and each word from the master set as further columns. Each row
will have a boolean for whether or not the word is present in the description
To do:
    Add matrix creation - think that this should be two different modules - 
    one to create the dataframe, in a similar way to the mast_list and one to
    create the boolean matrix. 
"""

import os
import argparse
import pandas
import ipdb
import numpy
import scipy.sparse
import pickle
from sklearn.externals import joblib

from ncvo_s2ds_2015.helpers import helpers

def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    
  # this is the default input directory if nothing is passed
  INPUT_MS = os.path.join("..", "..", "data", "features", "master_set.csv") 
  INPUT_DF = os.path.join("..", "..", "data", "features", "data_frame.csv") 
  OUTPUT_FILE = os.path.join("..", "..", "data", "features", "sparse_matrix.pkl") 


  parser.add_argument('--im', 
                      '--master-set-file', 
                      type=str, 
                      dest='inputMS', 
                      default=INPUT_MS, 
                      help='Master set of all words to be processed to boolean matrix')

  parser.add_argument('--id', 
                      '--data-frame-file', 
                      type=str, 
                      dest='inputDF', 
                      default=INPUT_DF, 
                      help='Data frame processed by "create_data_frame"')
   
  parser.add_argument('-o', 
                      '--output-file', 
                      type=str, 
                      dest='output_file', 
                      default=OUTPUT_FILE, 
                      help="File to save the sparse matrix. Defaults to '%(default)s'")


  args = parser.parse_args()
      
  master_in = pandas.read_csv(args.inputMS, encoding="ISO-8859-1", header=-1)
  data_in = pandas.read_csv(args.inputDF, encoding="ISO-8859-1")

  
  numrows = len(data_in)
  numcols = len(master_in)

  rowind = []
  colind = []
  data=[]

  for i in range(len(data_in[:3])):
    thisrow = data_in.iloc[i,2:]
    valid = thisrow.dropna()
    validset=set(list(valid))
    for setelem in validset:
       master_set_ind =  master_in.loc[master_in[0]==setelem][0].index[0]
       rowind.append(i)
       colind.append(master_set_ind)
       data.append(1)
  
  import ipdb; ipdb.set_trace()
  sparse_mat = scipy.sparse.coo_matrix((numpy.array(data),(numpy.array(rowind), numpy.array(colind))), shape=(numrows, numcols))
  
  # save the matrix
  helpers.ensure_dir(os.path.dirname(args.output_file))
  joblib.dump(sparse_mat, args.output_file)
  
  # len(df_in.columns)  
  # df_in.iloc(5)
  
  ipdb.set_trace()   
  
  
  # for loop here through the rows
  for x in range(0, len(df_in)):
      
      # for loop through cells in row - starting from 3rd? - where master set words start
      for y in range(2, len(df_in.columns)):
          
          print('x', x)
          print('y', y)

            
          token = df_in.iloc[x,y]
          #token = 'fuck'
          
          this_cell = [ 1 if token in master_set else 0]# for w in token ]
       

          #df_in.iloc[:1] #access row one
          
          if this_cell == 1:
              master_cols.ix[x, token] = 1
          
  

  # TODO:
  # - lematization
  # - removal of stopwords
  # - removal of weird numbers that are at the beginning
  # - removal of dulicates after stematizing / lemmatizing
  # - saving
>>>>>>> Stashed changes

if __name__ == "__main__":
  main()
