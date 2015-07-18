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




with open(file, 'rb') as csvfile:
    dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters=";,")
    csvfile.seek(0)
    reader = csv.reader(csvfile, dialect)

    print reader[0]
    
    for line in reader:
        print line





print data

print data[0]

print data[1]

for line in data:
    print line




xList = []
labels = []
names = []
firstLine = True
for line in data:
    if firstLine:
        names = line.strip().split(";")
        print names
        firstline = False
    else:
        row = line.strip().split(";")
        labels.append(float(row[-1]))
        row.pop()
        floatRow = [float(num) for num in row]
        xList.append(floatRow)
        


indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3 == 0]
xListTrain = [xList[i] for i in indices if i%3 != 0]
labelsTest = [labels[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]


attributeList = []
index = range(len(xList[1]))
indexSet = set(index)
indexSeq = []
oosError = []


for i in index:
    attSet = set(attributeList)
    attTrySet = indexSet - attset
    attTry = [ii for ii attTrySet]
    errorList = []
    attTemp = []
    
    for iTry in attTry:
        attTemp = [] + attibuteList
        attTemp.append(iTry)
        xTrainTemp = xattrSelect(xListTrain, attTemp)
        xTestTemp = xattrSelect(xListTest, attTemp)
        xTrain = numpy.array(xTrainTemp)
        yTrain = numpy.array(labelsTrain)
        xTest = numpy.array(xTestTemp)
        yTest = numpy.array(labelsTest)
        
        wineQModel = linear_model.LinearRegression()
        wineQModel = fit(xTrain, yTrain)
        rmsError = numpy.linalg.norm((yTest-wineQModel.predict(xTest)),
                                     2)/sqrt(len(yTest))
        errorList.append(rmsError)
        attTemp = []
        
    iBest = numpy.argmin(errorList)
    attributeList.append(attTry[iBest])
    oosError.append(errorList[iBest])
    

print("Out of sample error versus attribute set size" )
print(oosError)
print("\n" + "Best attribute indices")
print(attributeList)
namesList = [names[i] for i in attributeList]
print("\n" + "Best attribute names")
print(namesList)
    

# Error vs. # Attributes
x = range(len(oosError))
plt.plot(x,oosError, 'k')
plt.xlabel('# Attributes')
plt.ylabel('Error (RMS)')
plt.show()





indexBest = oosError.index(min(oosError))
attributesBest = attibuteList[1:(indexBest+1)]


xTrainTemp = xattrSelect(xListTrain, attributesBest)
xTestTemp = xattrSelect(xListTest, attributesBest)
xTrain = numpy.array(xTrainTemp)
xTest = numpy.array(xTestTemp)




wineQModel = linear_model.LinearRegression()
wineQModel.fit(xTrain, yTrain)
errorVector = yTest - wineQModel.predict(xTest)
plt.hist(errorVector)
plt.xlabel("Bin boundaries")
plt.ylabel("Counts")
plt.show()



plt.scatter(wineQModel.predict(xTest), yTest, s=100, alpha=0.10)
plt.xlabel("Predicted Taste Score")
plt.ylabel("Actual Taste Score")
plt.show()

























    























def confusionMatrix(predicted, actual, threshold):
    if len(predicted) != len(actual): return -1
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    
    for i in range(len(actual)):
        if actual[i] > 0.5:
            if predicted[i] > threshold:
                tp += 1.0
            else:
                fn += 1.0
        else:
            if predicted[i] < threshold:
                tn += 1.0
            else:
                fp += 1.0
    rtn = [tp, fn, fp, tn]
    return rtn
    
    
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")

target_url = ("C:\Users\craig\OneDrive\Machine Learning in Python\sonar.all-data")

with open(target_url) as f:
    data = f.readlines()
    

xList = []
labels = []
for line in data:
    row = line.strip().split(",")
    if (row[-1] == 'M'):
        labels.append(1.0)
    else:
        labels.append(0.0)
    row.pop()
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


# check shapes to see what they look like
print "Shape of xTrain array "
print xTrain.shape

print "Shape of yTrain array "
print yTrain.shape


print "Shape of xTest array "
print xTest.shape

print "Shape of yTest array "
print yTest.shape

# train linear regression model
rocksVMinesModel = linear_model.LinearRegression()
rocksVMinesModel.fit(xTrain, yTrain)

# predictions on in-sample error
trainingPredictions = rocksVMinesModel.predict(xTrain)
print "Some values predicted by model "
print trainingPredictions[0:5]
print trainingPredictions[-6:-1]


confusionMatTrain = confusionMatrix(trainingPredictions,yTrain,0.5)
tp = confusionMatTrain[0]
fn = confusionMatTrain[1]
fp = confusionMatTrain[2]
tn = confusionMatTrain[3]


print "tp = " + str(tp)
print "fn = " + str(fn)
print "tp = " + str(fp)
print "tn = " + str(tn)
print "\n"


# generate predictions on out-of-sample data
testPredictions = rocksVMinesModel.predict(xTest)

# generate confusion matrix
conMatTest = confusionMatrix(testPredictions, yTest, 0.5)

tp = conMatTest[0]
fn = conMatTest[1]
fp = conMatTest[2]
tn = conMatTest[3]


print "tp = " + str(tp)
print "fn = " + str(fn)
print "tp = " + str(fp)
print "tn = " + str(tn)
print "\n"


# generate ROC curve for in-sample
fpr, tpr, thresholds = roc_curve(yTrain, trainingPredictions)
roc_auc = auc(fpr, tpr)
print "AUC for in-sample ROC curve: %f" % roc_auc


# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
pl.plot([0, 1], [0, 1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])

pl.xlabel("False Pos Rate")
pl.ylabel("True Pos Rate")
pl.title("In sample ROC rocks vs mines")
pl.legend(loc="lower right")
pl.show()



# generate ROC curve for out of sample
fpr, tpr, thresholds = roc_curve(yTest, testPredictions)
roc_auc = auc(fpr, tpr)
print "AUC for out-of-sample ROC curve: %f" % roc_auc


# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
pl.plot([0, 1], [0, 1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])

pl.xlabel("False Pos Rate")
pl.ylabel("True Pos Rate")
pl.title("Out-of-Sample ROC rocks vs mines")
pl.legend(loc="lower right")
pl.show()



























