#!/usr/bin/python

import nltk
import string

def plot_frequency(word_list):
  """
  Plotting the frequency of most common words 
  """
  fd = nltk.FreqDist(word_list)
  fd.plot()
  fd.plot(50, cumulative=True)
  fd.most_common(12)   
  return fd 

def jaccard_similarity(word_list1, word_list2):
  """Measure of similairty of 1 bags of words   
  J(A, B)= (A and B)/(A or B)
  """
  intersection = set(word_list1).intersection(set(word_list2))
  union = set(list1).union(set(list2))
  return len(intersection)/len(union)   