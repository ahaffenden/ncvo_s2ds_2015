# ncvo-s2ds-2015

#Building text classifiers for the NCVO Almanac
=============================

This is the source code of a project done for the National Council for Volunteering Organizations (NCVO), UK. The project was done in the framework of the Science to Data Science (S2DS) ‘school ‘in London, UK, during August-September 2015. 
The project was done by the team of 4: Ivana Cingowska,  Austin Haffenden, Anna M. Kedzierska and Graeme Salter with great help on the side of the NCVO team.

Unfortunately, the data cannot be shared publicly.

##Step 1: Text processing
=======================

The script to use for text processing is "create_data_frame". This essentially creates a data structure (data frame/2D matrix). It serves as an input to both the feature extraction method (bag-of-words creation) and the classification if Step 2 is skipped (see below for details). Output is a csv file with the processed text. 

Exemplary command line:
```
./create_data_frame.py -s --sw --th 
```
Stemmatizes, removes stop words then tokenizes the input file according to the directory structure (one cell per section of the hierarchy). Output file will be saved in the default location with the default filename. 

Command line flags:
```
	-i 	input file from the data folder. Currently defaults to 'iTrain_extra_lastitem_mod_new.csv'
      ---lan	If set language is determined and non-english items removed. Boolean (default False). 
	-l	If set verbs will be lemmatized. Boolean (default = False)
       --la	If set all words will be lemmatized. Boolean (default = False)
       --lc	If set all words set to lower case. Boolean (default = True)
       --rw	If set types of words from a user defined list will be removed. Accepts a list of types of words to be removed from list of ADJ, ADV, CNJ, DET, EX, FW, MOD, N, NP, NUM, PRO, P, TO, UH, V, VD, VG, VN, WH'). String (default = None)
	-s	If set all words will be stematized. Boolean (default = False)
       --sa	If set removes accents leaving only the original letter. Boolean (default = True)
       --sp	If set corrects spelling mistakes word by word taking the most likely correction. Boolean (default = True)
       --sd	Allows setting of an alternate dictionary for spell checking. File and path (default = big.txt)
```

Output file column names:
```
	type_class, source_class, 0, 1 … n (with n dependent on the # hierarchy levels)
```
##Step 2: Feature extraction
==========================

Command to generate the features. 
This is essentially the bag of words function adapted to allow tokenization of the hierarchy in different ways, it also creates the pickle data structure. 

E.g:
```
./create_bow_fm_dataframe.py -i ../../data/features/words/none_iTrain.csv -o none_iTrain_stri.p
```

This call uses default flags. It takes in the output data frame saved in *none_iTrain.csv* created in Step 1. It then adds the charity type and name to the description list and then creates a bag of words from the hierarchical items. 
 
Command line flags:
```
       --ad	This adds charity type and name to the description. Boolean (default = True)
	-i	Dataframe to be processed to a Bag of words. String (default = '../../data/features/hierarchical/new/dframe_iTrain.csv.' )
       --lc	If set will extract the n last cells from the dataframe according to the input number. For use with source_class it needs a hierarchically tokenized dataframe. Int (default = 0)
       --hi	If set spaces will be removed from within hierarchical items. Boolean (default = False)
       --wl	Do not concatonate the last hierarchical item so that individual words are kept. Boolean (default = True)
       --od	Directory to save the output. String (default = '../../data/features')
	-o	Filename of output file. Pickle (.p format) (default = 'temp/data_stri_Please_rename_or_delete_me.p')
```
Output is a pickle (*.p*) data structure (= *.pkl*)

##Step 3: Classification
======================

The script to use for running a full pipeline of the code including bag-of-words or TFIDF feature extraction and classification, is *run_pipeline_wordfeats.py*. In the example below, the classification is performed on TFIDF features, using decision tree classifier and for the first classification task (income type) ::

```
./run_pipeline_wordfeats.py -i data/features/iTrain_stri.pkl --ti -c decision-tree --cat income-type --od data/res/iTrain_typeres_decision_tree.csv
```

The script will print the accuracy of the prediction and will output a *.csv* file with the original and the predicted classes for all the samples.

The default classifier values in this script are the ones that work the best based on the grid parameter search that we performed. If you want, you can run your own grid parameter search using the script ``gridsearch_classifiers_wordfeats.py``. To run the grid parameter search for TFIDF features and decision tree and for the first classification tasks (income type), run the following command:
```
./gridsearch_classfiers_wordfeats.py -i data/features/iTrain_stri.pkl --ti -c decision-tree --cat income-type --od data/gridres/iTrain_gridres_decision_tree.pkl
```

The script will print the parameters and the accuracy for the best classifier and will save them in a pickle file. This file can be reused in the *run_pipeline_wordfeats* script, by specifying the *-g* option.

To see all the options for the classification or grid search script, just type *—help* at the command line.

##Step 4: Computing joined accuracy
=================================

Joined accuracy for the type and source of income can be computed using the following command::
```
./typesource_joined_accuracy.py --io data/features/iTrain_stri.pkl --it data/res/iTrain_typeres_decision_tree.csv --is data/res/iTrain_sourceres_decision_tree.csv -o data/res/iTrain_joined_res.csv
```

To see all the options for the classification or grid search script use *—help* at the command line.  

Step 5: Plotting the results
============================

You can plot confusion matrix showing which labels were most frequently confused by the algorithm on some classification task using the following command (for income type):
```
./plot_confusion_matrices.py -i data/res/iTrain_typeres_decision_tree.csv --cat income-type
```

