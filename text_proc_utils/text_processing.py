#!/usr/bin/python

import nltk
#import ipdb
#import re
import unicodedata
from nltk.tokenize import RegexpTokenizer
#import langid
"""
Functions to exclude common words + some others if we want
"""

def text_tokenize_lower(word_list):
  """ tokenizes to lower case - uses the class string
       might not be needed 
  """
  tokenize = lambda doc: doc.lower().split(" ")
  word_list_token = tokenize(word_list)
  return word_list_token

def text_tokenize(word_list):
  """ tokenize  
  """
  word_list_token = nltk.word_tokenize(word_list)
  return word_list_token
  
def tokenize_on_hyphen(word_list):
     
  word_list_token_hy = nltk.regexp_tokenize(str(word_list),  r'\s[-]\s', gaps=True)
  #s3 = nltk.regexp_tokenize(word_list, r'[-]\s*', gaps=True)
  return word_list_token_hy

def keep_only_specified_tags(tokenized_word_list, list_of_tags):
  """Keeps only specified tags in the input list
     list_of_tags is a list of desired tags, e.g. ['VBP', 'NN']
    ADJ  adjective new, good, high, special, big, local
    ADV adverb  really, already, still, early, now
    CNJ conjunction and, or, but, if, while, although
    DET determiner  the, a, some, most, every, no
    EX  existential there, there's
    FW  foreign word  dolce, ersatz, esprit, quo, maitre
    MOD modal verb  will, can, would, may, must, should
    N noun  year, home, costs, time, education
    NP  proper noun Alison, Africa, April, Washington
    NUM number  twenty-four, fourth, 1991, 14:24
    PRO pronoun he, their, her, its, my, I, us
    P preposition on, of, at, with, by, into, under
    TO  the word to to
    UH  interjection  ah, bang, ha, whee, hmpf, oops
    V verb  is, has, get, do, make, see, run
    VD  past tense  said, took, told, made, asked
    VG  present participle  making, going, playing, working
    VN  past participle given, taken, begun, sung
    WH  wh determiner who, which, when, what, where, how
  """
  tokenized_word_list = nltk.pos_tag(tokenized_word_list) 
  newtext_tags_selected = []
  for (word, tag) in tokenized_word_list:
    if(tag in list_of_tags):
      newtext_tags_selected += [(word)] 

  return newtext_tags_selected

def exclude_stop_words(word_list):
  """Cleans the input list from stopwords. 
  """

  from nltk.corpus import stopwords
  stopwords = stopwords.words('english')
  newtext_no_stopwords= [w for w in word_list if w not in stopwords]

  return newtext_no_stopwords

def lemmatize(word_list, pos='v'):
  """ Lemmatizes the word_list

  Input: 
    word_list - list of words to be cleaned
    pos - the type of words to be lematized
  """
  wnl = nltk.WordNetLemmatizer()
  tokenizer = RegexpTokenizer(r'\w+')
  for x in range(0, len(word_list)):
      
  #tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
  #word_tokens = nltk.word_tokenize(str(word_list))
      word_tokens = tokenizer.tokenize(str(word_list[x]))
  #re.findall(r"[\w']+", str(word_tokens))
      word_tokens_lem = [wnl.lemmatize(w, pos=pos) for w in word_tokens]#word_tokens]
      sep = " "
      word_list[x] = sep.join(word_tokens_lem)
  return word_list
  
def lemmatizeall(word_list):
  """ Lemmatizes the word_list passing through each type of word

  Input: 
    word_list - list of words to be cleaned
    
    pos options: ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
  """
  word_types = "v", "a", "n", "s", "r"
  #print(word_types)
  #ipdb.set_trace() 
  wnl = nltk.WordNetLemmatizer()
  
  tokenizer = RegexpTokenizer(r'\w+')
  for x in range(0, len(word_list)):   
      
      word_tokens = tokenizer.tokenize(str(word_list[x]))
      word_tokens_lem = word_tokens
      for i in range(0, len(word_types)):
      
          pos = word_types[i]      
          word_tokens_lem = [wnl.lemmatize(w, pos=pos) for w in word_tokens_lem]
          
      sep = " "
      word_list[x] = sep.join(word_tokens_lem)
   
          #print(i)
  return word_list #[wnl.lemmatize(w, pos=pos) for w in word_list]  


def stematize(word_list):
  """ Stematizes the word_list
  """
  from nltk.stem.porter import PorterStemmer
  porter_stemmer = PorterStemmer()
  #ipdb.set_trace() 
  tokenizer = RegexpTokenizer(r'\w+')
  for x in range(0, len(word_list)):
      
  #tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
  #word_tokens = nltk.word_tokenize(str(word_list))
      word_tokens = tokenizer.tokenize(str(word_list[x]))
      word_tokens_stem = [porter_stemmer.stem(w) for w in word_tokens]  
      sep = " "
      word_list[x] = sep.join(word_tokens_stem)

  return word_list

def make_lower(word_list):
   """ Change to lower case letters
   """
   words_lc = list(map(str.lower, word_list)) 
   return words_lc

def make_upper(word_list):
   """ Change to upper case letters
   """
   words_uc = list(map(str.upper, word_list)) 
   return words_uc

def top_bigrams(word_list):
   """ Bigrams are words that are commonly found together
   """
   bigram_measures = nltk.collocations.BigramAssocMeasures()
   finder = nltk.collocations.BigramCollocationFinder.from_words(word_list)
   word_list_bigrams = finder.nbest(bigram_measures.pmi, 10)
   return word_list_bigrams

def top_trigrams(word_list):
   """ Trigrams are 3-tuples that are commonly found together
   """
   trigram_measures = nltk.collocations.TrigramAssocMeasures()
   finder = nltk.collocations.TrigramCollocationFinder.from_words(word_list)
   word_list_trigrams = finder.nbest(trigram_measures.pmi, 10)
   return word_list_trigrams

def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """

    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
           
    return str(text)
    
def language(word_list):
    """
    Determine the language of input text
    param = list of text strings
    returns = language class and probability of it being that language
    """
    langid.set_languages(['en', 'cy'])
    langid.rank
#    langid.set_languages(['en','cy'])    
    lang=[]  
    for x in range(0, len(word_list)):
        langval = langid.rank(word_list[x])
#        print(langval)
#        lang = lang + [langval[0]]    
        top_rank=langval[0]
        if top_rank[0] == 'en' : 
            lang = lang + [float(top_rank[1])]
        else:
            print(langval)
            print(word_list[x])
            eng_rank=langval[1]
            lang = lang + [float(eng_rank[1])]            
#    print(lang)
    return lang

  
def tfidf(data_frame):
  """ performs tf-idf on runs a spell checker on word_list
  """
  return data_frame