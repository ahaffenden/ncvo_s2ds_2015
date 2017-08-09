#! /usr/bin/env python
#Created on Fri Aug  7 17:10:28 2015
#@author: graeme
"""
This script creates a dataframe in a similar way to 'create_master_list' except 
that the type and source (if income) as the first and second column and each 
from the description in the final colun as a list. This dataframe is then used 
in the create_boolean_matrix module for use as input to ML functions. 
To do:
    Read in input file
    assign it to the relevant columns, Type, source, description (can add more later)
    carry out processing 
    output data product
"""
import os
import argparse
import pandas
import ipdb
import nltk
import numpy

from ncvo_s2ds_2015.text_proc_utils import text_processing
from ncvo_s2ds_2015.helpers import helpers

def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    
  # this is the default input directory if nothing is passed
  INPUT_FILE = os.path.join("..", "..", "data", "iTrain.csv") 
  OUTPUT_FILE = os.path.join("..", "..", "data", "features", "data-frame.csv") 
    
  parser.add_argument('--i', 
                      '--input-file', 
                      type=str, 
                      dest='inputFile', 
                      default=INPUT_FILE, 
                      help='File to be processed to boolean matrix')

  parser.add_argument('-l', 
                      '--lemmatize', 
                      dest='lemmatize', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, verbs will be lemmatized')
                      
  parser.add_argument('-la', 
                      '--lemmatizeall', 
                      dest='lemmatizeall', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, all words will be lemmatized')
                      
  parser.add_argument('-lc', 
                      '--lower-case', 
                      dest='lowerCase', 
                      action='store_true', 
                      default=True, 
                      help='Boolean - Defaults to converted all to lower-case')
                      
  parser.add_argument('-rw', 
                      '--remove-words', 
                      dest='removeWords', 
                      action='store_true', 
                      default=None, 
                      help='Accepts a list of types of words to be removed e.g. ...')

  parser.add_argument('-s', 
                      '--stematize', 
                      dest='stematize', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, all words will be stematized')    
  
  parser.add_argument('--sa', 
                      '--strip-accents',  
                      dest='stripAccents', 
                      action='store_true', 
                      default=False,  
                      help="Removes accents on letters replacing them with just the letter itself")

  parser.add_argument('--sp', 
                      '--spelling-corrector',  
                      dest='spellCorrect', 
                      action='store_true', 
                      default=False,  
                      help="Correct spelling mistakes word by word, just taking the most likely correction")
      
  parser.add_argument('--sw', 
                      '--stop-words', 
                      dest='stopWords', 
                      action='store_true', 
                      default=False, 
                      help='Removes the most common words, "stop words", from the text')

  parser.add_argument('-t', 
                      '-tokenize', 
                      dest='token', 
                      action='store_true', 
                      default=False, 
                      help='Tokenizes text to individual words')                      

  parser.add_argument('--ta', 
                      '--alpha-numeric', 
                      dest='alphaNumeric', 
                      action='store_false', 
                      default=True, 
                      help='Boolean - If NOT set file will be tokenized and non alpha-numeric words left in. Default is TRUE')
     
  parser.add_argument('--th', 
                      '--token-hyphen', 
                      dest='tokenHyphen', 
                      action='store_true', 
                      default=False, 
                      help='Tokenizes text using the hierarchy structure')
                      
  parser.add_argument('-uc', 
                      '--upper-case', 
                      dest='upperCase', 
                      action='store_true', 
                      default=False, 
                      help='Boolean - If set, all words will be converted to upper-case')
                      
  parser.add_argument('-o', 
                      '--output-file', 
                      type=str, 
                      dest='output_file', 
                      default=OUTPUT_FILE, 
                      help="Directory to be used to save the created master set. Filename will be automatically created based on input flags. Defaults to something needsto go here")                      

    
  args = parser.parse_args()
      
  data_in = pandas.read_csv(args.inputFile, encoding="ISO-8859-1")
  
  #set column names? 
  words = pandas.DataFrame({'type':data_in.type_class, 
                           'class': data_in.source_class, 
                           'description': data_in.description}) 
  #set column names? 
    
  #print(words.head())
 
#=============================================================================
# specify data frame of the required length
#=============================================================================
 
#==============================================================================
#  # frame_len = 0
#   #for x in range(1, len(words)):
#       
#   #  temp_len = len(words.description[x].split())
#    # if temp_len > frame_len:
#         frame_len = temp_len
#         print(frame_len)
#         print(words.description[x].split())
# 
#==============================================================================  
  processed_data = pandas.DataFrame(data_in[['type_class', 'source_class']])
  
# function calls need to be edited to send the relevant columns (excluding type and source).
# it will require a restructuring of data types. Not sure if can use a dynamic data frame
# as don't know how many words in each row - also will be different depending on processing
  
  #tokenize the text either straight or keeping only alpha-numeric(default)
  if args.alphaNumeric:
      
    # keep just the alpha-numeric characters
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    word_list = list(map(tokenizer.tokenize, words.description))
    #print("alpha numeric only")    
    #print(word_list[1:20])      
    
#  else:
 #   word_list = list(map(nltk.word_tokenize, words.description))
    #print(word_list[1:20])   
   
  if args.tokenHyphen:
      #print("STEMATIZE")
      word_list = list(map(text_processing.tokenize_on_hyphen, words.description))
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
          
  if args.lemmatizeall:    
     #print("LEMMATIZE ALL")
      word_list = list(map(text_processing.lemmatizeall, word_list))
     #print(word_list[1:20])      
      
  if args.lemmatize:    
      #print("LEMMATIZE")
      word_list = list(map(text_processing.lemmatize, word_list))
      #print(word_list[1:20])      

  if args.removeWords:
      print("REMOVE WORDS")
      #needs function in text_processing 
      
  if args.stematize:
      #print("STEMATIZE")
      word_list = list(map(text_processing.stematize, word_list))
      #print(word_list[1:20])    
                
  if args.stopWords:
      #print("STOP WORDS")
      word_list = list(map(text_processing.exclude_stop_words, word_list))
      #print(word_list[1:20])      
      
    
    
  wl_df = pandas.DataFrame(word_list)
  frames = [processed_data, wl_df]
  
  output_df = pandas.concat(frames, axis = 1)

  #print(output_df[1:20])

  helpers.ensure_dir(os.path.dirname(args.output_file))
  output_df.to_csv(args.output_file, index = False) # write to file, but don't give row names
 

if __name__ == "__main__":
  main()
