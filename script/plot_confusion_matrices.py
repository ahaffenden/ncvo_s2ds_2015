#!/usr/bin/python

# input arguments specify the labels of the plots:
# itype, isource, etype 
#USAGE:
# python plot_confusion_matrices.py itype ../../../predicted_labels_ensemble.pkl fig_name
#
#NOTE: for now to be used with a bash script: process_pickles.sh

import pickle
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import fileinput

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import sys

class_type = sys.argv[1]
pickle_name = sys.argv[2]
out_fig_name = sys.argv[3]


def plot_classification_report(cr, fig_name, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat[:,1], interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Precision')
    plt.savefig(fig_name)# save the figure to file
    plt.close()

def plot_classification_report_prec(cr, fig_name,with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)
    a=[]
    for i in range(0,len(plotMat)):
        v = [plotMat[i][1]]
        a.append(v)
       
    plt.imshow(a, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    x_tick_marks = np.arange(1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['Precision'])#, rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    #plt.xlabel('Precision')
    plt.savefig(fig_name)# save the figure to file
    plt.close()

def my_custom_loss_func(X):
    diff = np.abs(X[0] - X[1]).max()
    return np.log(1 + diff)

def plot_confusion_matrix(cm, fig_name, target_names, xlab, ylab, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel(ylab)
    plt.xlabel(xlab) #plt.figure()
    plt.show()
    
if __name__ == "__main__":

    X = pickle.load(open(pickle_name, 'rb'))

    if class_type == 'itype':
            print('analysing income type')
            target_names = ['IC', 'IG', 'IGI', 'IV']
    elif class_type == 'isource':
            print('analysing income source')    
            target_names = ['100','110','121','125','132','140','161','162','163','171','172','175','180','200','300','330','500','600','620','700','710','720','730']
    else:
            print('analysing expenditure type')
            target_names = ['EC','EF', 'EFF', 'EFI', 'EFV','EG','EM','EMA','EMO']
    # change floats to integers
    
   if class_type == 'isource':
        x0temp = X[0][:-1]
        X[0] = [str(round(float(a))) for a in x0temp]
        x1temp = X[1][:-1]
        X[1] = [str(round(float(a))) for a in x1temp]



    
    classificationReport = classification_report(X[0], X[1], target_names=target_names)

    acc_score = accuracy_score(X[0],X[1])
    
    cm = confusion_matrix(X[0], X[1], target_names)

    np.set_printoptions(precision=2)

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print('Normalized confusion matrix')
    
    plot_confusion_matrix(cm_normalized, out_fig_name, target_names,  'Predicted classes', 'Manually assigned classes')

    np.savetxt(out_fig_name + '.out' ,cm_normalized, fmt='%.4e ')
    plot_classification_report_prec(classificationReport, out_fig_name + 'class_rep')
  


    