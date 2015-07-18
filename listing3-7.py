# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:23:10 2015

@author: craig
"""

import urllib2
import numpy
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as plt
import csv

file = 'sonar.all-data'

with open(file, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    

xList = []
labels = []
for line in data:
#    for element in line:
#        print element
    ln = len(line)
#    print len(line)
#    print line[-1]
#    print line[-1] == 'M'
#    print line[ln-1] == 'M'
    if (line[ln-1] == 'M'):
        labels.append(1.0)
    else:
        labels.append(0.0)
    linenum = line[0:(ln-2)]
    floatrow = [float(num) for num in linenum]
    xList.append(floatrow)
    print floatrow

    
indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3 == 0]
xListTrain = [xList[i] for i in indices if i%3 != 0]
labelsTest = [labels[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]

xTrain = numpy.array(xListTrain)
yTrain = numpy.array(labelsTrain)
xTest = numpy.array(xListTest)
yTest = numpy.array(labelsTest)

alphaList = [0.1**i for i in [-3, -2, -1, 0, 1, 2, 3, 4, 5]]



aucList = []
for alph in alphaList:
    rocksVMinesRidgeModel = linear_model.Ridge(alpha=alph)
    rocksVMinesRidgeModel.fit(xTrain, yTrain)
    fpr, tpr, thresholds = roc_curve(yTest,rocksVMinesRidgeModel.predict(xTest))
    roc_auc = auc(fpr, tpr)
    aucList.append(roc_auc)



print("AUC alpha")
for i in range(len(aucList)):
    print(aucList[i], alphaList[i])


#plot auc values versus alpha values
x = [-3, -2, -1, 0,1, 2, 3, 4, 5]
plt.plot(x, aucList)
plt.xlabel('-log(alpha)')
plt.ylabel('AUC')
plt.show()

#visualize the performance of the best classifier
indexBest = aucList.index(max(aucList))
alph = alphaList[indexBest]
rocksVMinesRidgeModel = linear_model.Ridge(alpha=alph)
rocksVMinesRidgeModel.fit(xTrain, yTrain)


#scatter plot of actual vs predicted
plt.scatter(rocksVMinesRidgeModel.predict(xTest),
yTest, s=100, alpha=0.25)
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.show()









