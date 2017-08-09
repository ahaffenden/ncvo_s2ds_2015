#! /usr/bin/env python
"""
Created on Wed Aug 19 12:12:29 2015

@author: greybeard
"""

import os
#import inspect
import ipdb
#import nltk
#import numpy
import argparse
import pandas
import pickle 
import math

#from sklearn.feature_extraction.text import CountVectorizer
from ncvo_s2ds_2015.helpers import helpers

def main():
    
  # to parse the arguments that are passed to main
  parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    
  # this is the default input directory if nothing is passed
  # this default file still contains I and IO values so needs updating
  INPUT_FILE = os.path.join("..", "..", "data", "features", "hierarchical", "new", "dframe_iTrain.csv")
  OUTPUT_DIR = os.path.join("..", "..", "data", "features")
  OUTPUT_FILENAME = 'temp/data_stri_Please_rename_or_delete_me.p'
  
  parser.add_argument('--ad',
                      '--addData',
                      dest='addData',
                      action='store_false', 
                      default=True, 
                      help='adds charity type and name to description list')   
                      

  parser.add_argument('-i', 
                      '-input-file', 
                      type=str, 
                      dest='inputFile', 
                      default=INPUT_FILE, 
                      help='Dataframe to be processed to a Bag of words. Defaults to ../../data/features/data_frame_test.csv.' )
 
  parser.add_argument('--lc',
                       '--lastCells',
                       dest='lastCells',
                       type = int,
                       default=0,
                       help='If set will extract the n last cells from the dataframe according to the input number. For use with source_class it needs a hierarchically tokenized dataframe'
                       )  
  
  parser.add_argument('--hi',
                      '--hierarchy',
                      dest='hierarchy',
                      action='store_true', 
                      default=False, 
                      help='If set spaces will be removed from within hierarchical items')    
  
  
  parser.add_argument('--wl',
                      '--wordsForLast',
                      dest='wordsForLast',
                      action='store_true', 
                      default=False, 
                      help='Do not concatonate the last hierarchical item so that individual words are kept')   
                      
  parser.add_argument('--od', 
                      '-output-dir', 
                      type=str, 
                      dest='output_dir', 
                      default=OUTPUT_DIR, 
                      help="Directory to save the output")    

  parser.add_argument('-o', 
                      '-output-file', 
                      type=str, 
                      dest='outputFilename', 
                      default=OUTPUT_FILENAME, 
                      help="Filename of output file - must be a pickle.p format")                      
                      
  args = parser.parse_args()

  data_in = pandas.read_csv(args.inputFile, encoding="ISO-8859-15" )
  data_in_str = pandas.read_csv(args.inputFile, encoding="ISO-8859-15" ,dtype=str)

#  data_in = data_in[data_in.type_class != 'I']
#  data_in = data_in[data_in.type_class != 'IO']
 
  data_stri = data_in_str[['frID','type_class','source_class']]
  data_stri['description'] = 'Nan' 
  sep = " "
  if args.hierarchy:
    print ('in hierarchy separation mode')    
    token_data = data_in.iloc[0:,5:]
    for i in range(0, len(token_data)):
        row=token_data.iloc[i,0:]      
        for j in range(0, len(row)):
            token=row.iloc[j]
            if args.wordsForLast :
                if j < (len(row)-1) :
                    nextToken=row.iloc[j+1]
                    if isinstance(nextToken, float) : 
                        if not math.isnan(nextToken) :
                          if type(token) == str :
                            token=token.replace(" ", "")
                    else :
                        if type(token) == str :
                          token=token.replace(" ", "")
            else :    
                if type(token) == str :
                    token=token.replace(" ", "")
            row[j]=token
        token_data.iloc[i]=row
    data_in.iloc[0:,5:]=token_data#.iloc[0:,0:]

  if args.addData : 
      charity_type=data_in_str.iloc[0:,3]
      charity_name=data_in_str.iloc[0:,4]     
      
   
   
  if args.lastCells == 0: # effectively processing type_class
      print('last cells equals 0')
   
      token_data = data_in.iloc[0:,5:]
      i=0
      for row in range(0, len(token_data)):
          
          token_data_stri = sep.join(map(str, token_data.iloc[row,0:].dropna()))
          if args.addData :
              collapsedname = charity_name.iloc[row].replace(" ", "")
              collapsedtype = charity_type.iloc[row].replace(" ", "")
              all_description=sep.join([token_data_stri,collapsedname,collapsedtype])
              data_stri.iloc[row, 3] = all_description
              i=i+1
              if i % 100 == 0 : print('row %d'% i)
          else :
              
              data_stri.iloc[row, 3] = token_data_stri 

      
  else: # effectively processing for source class as this selects the last n 
        # segments of the hierachy - needs to be passed the hierachical dataframe
      #print('last cells equals ', lastCells)

      for row in range(0, len(data_in)):
          
          print(row)
          
          all_cells = list()
          for col in data_in.iloc[int(row), 5:]:
              #print("col: ", col)
              all_cells.append(col)
   
          all_cells = [x for x in all_cells if str(x) != 'nan']
        
          data_stri.iloc[int(row), 3] = sep.join(all_cells[len(all_cells)- args.lastCells:len(all_cells)])    

  helpers.ensure_dir(args.output_dir)

  data_stri.to_pickle(os.path.join(args.output_dir, args.outputFilename))

if __name__ == "__main__":
  main()

