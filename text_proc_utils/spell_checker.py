# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:21:35 2015
based on http://norvig.com/spell-correct.html
@author: graeme
"""

import re, collections
from nltk.tokenize import RegexpTokenizer
#import ipdb
def words(text): return re.findall('[a-z]+', str.lower(text)) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model
    
#dictwords = train(words(ref_words.read()))
##ref_words=open('../../data/big.txt', "r")
#alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word, alphabet):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word, dictwords, alphabet):
    return set(e2 for e1 in edits1(word, alphabet) for e2 in edits1(e1, alphabet) if e2 in dictwords)

def known(wordsx, dictwords): 
#    object.__hash__(wordsx)
#    print((dictwords))
    return set(w for w in wordsx if w in dictwords)

def correct(word, dictwords, alphabet):
    candidates = known([word], dictwords) or known(edits1(word, alphabet), dictwords) or known_edits2(word, dictwords, alphabet) or [word]
    return max(candidates, key=dictwords.get)
    
def correctall(wordlist, SPELL_CHECKER_PATH):
      ref_words=open(SPELL_CHECKER_PATH, "r")            
      dictwords = train(words(ref_words.read()))
#      print(type(dictwords))
      alphabet = 'abcdefghijklmnopqrstuvwxyz'
      #
      tokenizer = RegexpTokenizer(r'\w+')
      sep = " "
      i=0

      #split to item
      for x in range(0, len(wordlist)) :
          single_item = wordlist[x]
          for k in range(0, len(single_item)):
              high_sub_item=tokenizer.tokenize(str(single_item[k]))
#              for j in range(0, len(high_sub_item)):             
              CorrectedWords  = [correct(word, dictwords, alphabet) for word in high_sub_item]
#                  high_sub_item[j]= CorrectedWords
              single_item[k]=sep.join(CorrectedWords)    
     #combine string       
#       correctedWord  = [correct(y, dictwords, alphabet) for y in wordlist[x]]
#       wordlist[x]= correctedWord
          i=i+1
#          if i % 100 == 0 : print('row %d'% i)
#     wordlist = list(map(correct, wordlist))
          wordlist[x]=single_item
       
      return wordlist