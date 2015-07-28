# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:23:10 2015

@author: craig
"""


from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl

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
    
    

file = "sonar.all-data"
with open(file) as f:
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



























