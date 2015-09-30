__author__ = 'mike-bowles'

import urllib2
import numpy
import matplotlib.pyplot as plot
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from math import floor
import random


import csv

def getWineData(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return (data)


def wineData2ListLabelsNames(data):

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
            
    return (xList, labels, names)



file = "winequality-red.csv"
data = getWineData(file)
xList, labels, names = wineData2ListLabelsNames(data)

nrows = len(xList)
ncols = len(xList[0])

X = numpy.array(xList)
y = numpy.array(labels)
wineNames = numpy.array(names)

#take fixed holdout set 30% of data rows
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30, random_state=531)


#maximum number of models to generate
numTreesMax = 100

#tree depth - typically at the high end
treeDepth = 10

#initialize a list to hold models
modelList = []
predList = []

#number of samples to draw for stochastic bagging
bagFract = 0.75 # proportion of samples
nBagSamples = int(len(xTrain) * bagFract) # samples

for iTrees in range(numTreesMax):
    idxBag = []
    for i in range(nBagSamples):
        idxBag.append(random.choice(range(len(xTrain))))
    xTrainBag = [xTrain[i] for i in idxBag]
    yTrainBag = [yTrain[i] for i in idxBag]

    modelList.append(DecisionTreeRegressor(max_depth=treeDepth))
    modelList[-1].fit(xTrainBag, yTrainBag)

    #make prediction with latest model and add to list of predictions
    latestPrediction = modelList[-1].predict(xTest)
    predList.append(list(latestPrediction))


#build cumulative prediction from first "n" models
mse = []
allPredictions = []
for iModels in range(len(modelList)):

    #average first "iModels" of the predictions
    prediction = []
    for iPred in range(len(xTest)):
        prediction.append(sum([predList[i][iPred] for i in range(iModels + 1)])/(iModels + 1))

    allPredictions.append(prediction)
    errors = [(yTest[i] - prediction[i]) for i in range(len(yTest))]
    mse.append(sum([e * e for e in errors]) / len(yTest))

nModels = [i + 1 for i in range(len(modelList))]

plot.plot(nModels,mse)
plot.axis('tight')
plot.xlabel('Number of Models in Ensemble')
plot.ylabel('Mean Squared Error')
plot.ylim((0.0, max(mse)))
plot.show()

print('Minimum MSE')
print(min(mse))


#With treeDepth = 5
#     bagFract = 0.5
#Minimum MSE
#0.429310223079

#With treeDepth = 8
#     bagFract = 0.5
#Minimum MSE
#0.395838627928

#With treeDepth = 10
#     bagFract = 1.0
#Minimum MSE
#0.313120547589