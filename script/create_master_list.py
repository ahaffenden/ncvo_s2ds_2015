#! /usr/bin/env python
#Created on Fri Aug  7 17:10:28 2015
#@author: graeme
"""
This script creates a master set (a set of all the unique words in file), 
after user specified text processing. 
To do - needs function call for auto spell correct
        needs file encoding to be checked as some characters read in incorrectly and split words
"""

import os
#import inspect
import argparse
import pandas
import ipdb
import nltk
import numpy


from ncvo_s2ds_2015.text_proc_utils import text_processing
from ncvo_s2ds_2015.text_proc_utils import spell_checker
from ncvo_s2ds_2015.helpers import helpers

def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    
  # this is the default input directory if nothing is passed
  INPUT_FILE = os.path.join("..", "..", "data", "test.csv") #iTrain.csv
  FOLD_FILE = os.path.join("..", "..", "data", "folds", "iTrain_fold.csv") #iTrain.csv
  OUTPUT_DIR = os.path.join("..", "..", "data", "test") 
  SPELL_CHECKER_PATH = os.path.join("..","..","data","big.txt")

  parser.add_argument('-i', 
                      '-input-file', 
                      type=str, 
                      dest='inputFile', 
                      default=INPUT_FILE, 
                      help='File to be processed to a master set. Defaults to iTrain.csv. File must be saved in data and argument structured as - ../../data/yourfilename.csv' )

  parser.add_argument('---lan', 
                      '---language', 
                      dest='language', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, language will be determined and non-english items will be removed')

  parser.add_argument('-l', 
                      '-lemmatize', 
                      dest='lemmatize', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, verbs will be lemmatized')
                      
  parser.add_argument('--la', 
                      '--lemmatizeall', 
                      dest='lemmatizeall', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, all words will be lemmatized')
                      
  parser.add_argument('--lc', 
                      '--lower-case', 
                      dest='lowerCase', 
                      action='store_false', 
                      default=True, 
                      help='Boolean - Defaults to converted all to lower-case')
                      
  parser.add_argument('--rw', 
                      '--remove-words', 
                      dest='removeTags',
                      nargs= '+',
                      #action='store_true', 
                      default=None, 
                      help='Accepts a list of types of words to be removed from list of ADJ, ADV, CNJ, DET, EX, FW, MOD, N, NP, NUM, PRO, P, TO, UH, V, VD, VG, VN, WH')

  parser.add_argument('-s', 
                      '-stematize', 
                      dest='stematize', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, all words will be stematized')    
                      
  parser.add_argument('--sa', 
                      '--strip-accents',  
                      dest='stripAccents', 
                      action='store_false', 
                      default=True,  
                      help="Removes accents on letters replacing them with just the letter itself")

  parser.add_argument('--sp', 
                      '--spelling-corrector',  
                      dest='spellCorrect', 
                      action='store_false', 
                      default=True,  
                      help="Correct spelling mistakes word by word, just taking the most likely correction")
  
  parser.add_argument('--sw', 
                      '--stop-words', 
                      dest='stopWords', 
                      action='store_false', 
                      default=True, 
                      help='Removes the most common words, "stop words", from the text')
                      
#  parser.add_argument('-t', 
#                      '-tokenize', 
#                      dest='tokenize', 
#                      action='store_true', 
#                      default=False, 
#                      help='Tokenizes text to individual words')                      

  parser.add_argument('--ta', 
                      '--alpha-numeric', 
                      dest='alphaNumeric', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If NOT set file will be tokenized and non alpha-numeric words left in. Default is TRUE')
     
  parser.add_argument('--th', 
                      '--token-hyphen', 
                      dest='tokenHyphen', 
                      action='store_true', 
                      default=False, 
                      help='Tokenizes text using the directory structure from input file')

  parser.add_argument('--uc', 
                      '--upper-case', 
                      dest='upperCase', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, all words will be converted to upper-case')

  parser.add_argument('--fn', 
                      '--fold-number', 
                      dest='fold_number', 
                      default=0,
                      type=int, 
                      help="The fold number to be EXCLUDED when creating the master set (If there are N folds, N-1 folds are used for creating the master set). If 0, all the items will be considered . Defaults to '%(default)s'")

  parser.add_argument('--ff', 
                      '--fold-file', 
                      type=str, 
                      dest='foldFile', 
                      default=FOLD_FILE, 
                      help="Fold file containing the cross-fold validation indices. Defaults to '%(default)s'")
                      
  parser.add_argument('-o', 
                      '-output-dir', 
                      type=str, 
                      dest='output_dir', 
                      default=OUTPUT_DIR, 
                      help="Directory to be used to save the created master set. Filename will be automatically created based on input flags. Defaults to something needsto go here")                      

    
  args = parser.parse_args()
      
#  data_in = pandas.read_csv(args.inputFile, encoding="ISO-8859-1")
  data_in = pandas.read_csv(args.inputFile, encoding="ISO-8859-15" )
  words = pandas.DataFrame(data_in.description) # take just the description data
  
  # Filter the folds for master set training
  if args.fold_number != 0: # for creating the master set, keep just the words that DO NOT belong to the fold given as an argument
    cross_fold_indices = pandas.read_csv(args.foldFile, header=None)[0]
    words = words.loc[cross_fold_indices != args.fold_number]
 
  #Define word_list
  word_list=words.description
  
  #print(words.head())
  if args.language:
      #check that text is in english and separate 
    langval=text_processing.language(word_list)
#    for x in range(0, len(word_list)):
    good=numpy.where([x >= 0.8 for x in langval])
    bad=numpy.where([x < 0.8 for x in langval])
#    good=numpy.where([x=='en' for x in langval])
#    bad=numpy.where([x!='en' for x in langval])
    badwords = words.drop(words.index[good])
    words = words.drop(words.index[bad])
    word_list=words.description
    #
    #
    bad_output_fileName = "welsh_set.csv"
    bad_output_file = os.path.join(args.output_dir, bad_output_fileName)    
    helpers.ensure_dir(os.path.dirname(args.output_dir))
    badwords.to_csv(bad_output_file, index = False) # write to file, but don't give row names
    

  #tokenize the text either straight or keeping only alpha-numeric(default)
  if args.alphaNumeric:
      
    # keep just the alpha-numeric characters
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    word_list = list(map(tokenizer.tokenize, word_list))
    #print("alpha numeric only")    
    #print(word_list[1:20])      
    
  if args.tokenize:
          word_list = list(map(tokenizer.tokenize, word_list))
      
 #   word_list = list(map(nltk.word_tokenize, words.description))
    #print(word_list[1:20])      
  if args.tokenHyphen:
  #print("Tokenize on hyphen")
    word_list = list(map(text_processing.tokenize_on_hyphen , word_list))
    #print(word_list[1:20])      

  # lower case 
  if args.lowerCase:
    #print("LOWER CASE")
    word_list = list(map(text_processing.make_lower, word_list))      
    #print(word_list[1:20])
  
  # Upper case      
  if args.upperCase:
    #print("UPPER CASE")
    word_list = list(map(text_processing.make_upper, word_list))      
    #print(word_list[1:20])
    
  if args.stripAccents:
      #print("CORRECT SPELLING")
    i=0
    for x in range(len(word_list)) :
       correctedWords  = [text_processing.strip_accents(y) for y in word_list[x]]
       word_list[x]= correctedWords
       i=i+1
       if i % 100 == 0 : print('row %d'% i)
#    word_list = list(map(text_processing.strip_accents, word_list))
      #print(word_list[1:20])    

  if args.spellCorrect:
      #print("CORRECT SPELLING")
    word_list = spell_checker.correctall(word_list, SPELL_CHECKER_PATH)
      #print(word_list[1:20])     

  if args.removeTags:
      #print("REMOVE WORDS")
      #needs function in text_processing 
      word_list = list(map(text_processing.keep_only_specified_tags, word_list, args.removeTags))
      
  if args.stopWords:
      #print("STOP WORDS")
      word_list = list(map(text_processing.exclude_stop_words, word_list))
      #print(word_list[1:20])      
          
  if args.lemmatizeall:    
      #print("LEMMATIZE ALL")
      word_list = list(map(text_processing.lemmatizeall, word_list))
      #print(word_list[1:20])      
   
  if args.lemmatize:    
      #print("LEMMATIZE")
      word_list = list(map(text_processing.lemmatize, word_list))
      print(word_list[1:20])      

  if args.stematize:
      #print("STEMATIZE")
      word_list = list(map(text_processing.stematize, word_list))
      #print(word_list[1:20])       

  master_set = set()

  list(map(master_set.update, word_list)) # add unique values to the master set
  sorted(master_set)

  #ipdb.set_trace()
  
  helpers.ensure_dir(args.output_dir)

  #ipdb.set_trace()
  
  if args.fold_number!=0:
      output_fileName = "master_set_fold_%d.csv" % args.fold_number
  else:
      output_fileName = "master_set.csv"
  output_path = os.path.join(args.output_dir, output_fileName)    
  master_set_towrite = pandas.Series(numpy.array(list(master_set)))
  master_set_towrite.to_csv(output_path, index=False) # write to file, but don't give row names

if __name__ == "__main__":
  main()
