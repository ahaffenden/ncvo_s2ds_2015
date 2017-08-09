#!/usr/bin/python

# Test script
# Grid Search varying weights assigned to 3 chosen classifiers.

import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from sklearn import datasets
import pickle 

from ncvo_s2ds_2015.classifiers import ensemble_classifier 


#iris = datasets.load_iris()
#X, y = iris.data[:, 1:3], iris.target

np.random.seed(123)


#X = counted # sparse matrix input
  #X = tfidf
  #X = counted_bigr
  #y = data_str.iloc[:, 0]

X = pickle.load(open('../../data/features/hier_data.pkl', 'rb'))
X = X.toarray()
y = pickle.load(open('../../data/features/hier_labels.pkl', 'rb'))

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = DecisionTreeClassifier()

#clf3 = GaussianNB()


# pickling the best trained regressors!!!!!!!


label =['Logistic Regression', 'Random Forest', 'Decision Tree']

print(label)
df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'mean', 'std'))

i = 0
for w1 in range(0,3):
    for w2 in range(0,3):
        for w3 in range(0,3):

            if len(set((w1,w2,w3))) == 1: # skip if all weights are equal
                continue

            eclf = ensemble_classifier.EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[w1,w2,w3])
            scores = cross_validation.cross_val_score(
                                            estimator=eclf,
                                            X=X,
                                            y=y,
                                            cv=5,
                                            scoring='accuracy',
                                            n_jobs=1)

            df.loc[i] = [w1, w2, w3, scores.mean(), scores.std()]
            print("Accuracy: %0.5f (+/- %0.5f) w1=%d w2=%d w3=%d" % (scores.mean(), scores.std(), w1, w2, w3))
            i += 1

df.sort(columns=['mean', 'std'], ascending=False)
# printing out the results:
# w1, w2, w3, mean (averaged over k-folds), std
df.to_csv('../../data/grid_search_output/grid_search_ensemble.csv')


