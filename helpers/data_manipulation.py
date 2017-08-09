#! /usr/bin/env python
# Tue Aug 11 08:57:52 BST 2015
# Ivana Chingovska
"""
This script contains helper functions for data manupulation
"""

import scipy.sparse
import os
import numpy

def binary_sparse_matrix(data_in, master_in):
  """Method that creates a binary sparse matrix from the input dataframe file. Each row corresponds to a row in data_in. Each column corresponds to an element in the master_in"""

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
  
  sparse_mat = scipy.sparse.coo_matrix((numpy.array(data),(numpy.array(rowind), numpy.array(colind))), shape=(numrows, numcols))
  return sparse_mat
