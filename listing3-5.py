# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:23:10 2015

@author: craig
"""







import urllib2
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

#read data into iterable
target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = urllib2.urlopen(target_url)

xList = []
labels = []
names = []
firstLine = True
for line in data:
    if firstLine:
        names = line.strip().split(";")
        firstLine = False
    else:
        #split on semi-colon
        row = line.strip().split(";")
        #put labels in separate array
        labels.append(float(row[-1]))
        #remove label from row
        row.pop()
        #convert row to floats
        floatRow = [float(num) for num in row]
        xList.append(floatRow)














import urllib2
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

#read data into iterable
target_url = ("http://archive.ics.uci.edu/ml/machine-learningdatabases/wine-quality/winequality-red.csv")

data = urllib2.urlopen(target_url)


req = urllib2.Request(target_url)

try:
    data = urllib2.urlopen(req)
except urllib2.URLError as e:
    if hasattr(e,'reason'):
        print 'We failed to reach a server.'
        print 'Reason: ', e.reason
    elif hasattr(e,'code'):
        print 'The server couldn\'t fulfill the request.'
        print 'Error code: ', e.code
else:
    print 'Everything is fine.'




file = 'winequality-red.csv'

import csv
with open(file, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
    
xList = []
labels = []
names = []
firstLine = True
for line in data:
    if firstLine:
        names = line[0].strip().split(";") # [0] gets string, not list
        firstLine = False
    else:
        #split on semi-colon
        row = line[0].strip().split(";")
        #put labels in separate array
        labels.append(float(row[-1]))
        #remove label from row
        row.pop()
        #convert row to floats
        floatRow = [float(num) for num in row]
        xList.append(floatRow)




indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3 == 0]
xListTrain = [xList[i] for i in indices if i%3 != 0]
labelsTest = [labels[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]

xTrain = numpy.array(xListTrain)
yTrain = numpy.array(labelsTrain)
xTest = numpy.array(xListTest)
yTest = numpy.array(labelsTest)

alphaList = [0.1**i for i in [0,1,2,3,4,5,6]]

rmsError = []
for alph in alphaList:
    wineRidgeModel = linear_model.Ridge(alpha=alph)
    wineRidgeModel.fit(xTrain,yTrain)
    rmsError.append(numpy.linalg.norm((yTest-wineRidgeModel.predict(xTest)),2)/sqrt(len(yTest)))
    

print "RMS Error       alpha"
for i in range(len(rmsError)):
    print "%.4f    %.4f\n" % (rmsError[i], alphaList[i])
    
    
x = range(len(rmsError))
plt.plot(x, rmsError)
plt.xlabel('-log(alpha)')
plt.ylabel('Error (RMS)')
plt.show()



indexBest = rmsError.index(min(rmsError))
alph = alphaList[indexBest]
wineRidgeModel = linear_model.Ridge(alpha=alph)
wineRidgeModel.fit(xTrain, yTrain)
errorVector = yTest - wineRidgeModel.predict(xTest)
plt.hist(errorVector)
plt.xlabel("Bin Boundaries")
plt.ylabel("Counts")
plt.show()


plt.scatter(wineRidgeModel.predict(xTest), yTest, s=100, alpha=0.10)
plt.xlabel("Predicted Taste Score")
plt.ylabel("Actual Taste Score")
plt.show()

