__author__ = 'mike-bowles'

import csv
from sklearn.tree import DecisionTreeRegressor
import random
import matplotlib.pyplot as plot

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


def wineList2TrainTest(xList,labels,prop):
    
    nrows = len(xList)

    #take fixed test set 30% of sample
    random.seed(1)
    nSample = int(nrows * 0.30)
    idxTest = random.sample(range(nrows), nSample)
    idxTest.sort()
    idxTrain = [idx for idx in range(nrows) if not(idx in idxTest)]
    
    #Define test and training attribute and label sets
    xTrain = [xList[r] for r in idxTrain]
    xTest = [xList[r] for r in idxTest]
    yTrain = [labels[r] for r in idxTrain]
    yTest = [labels[r] for r in idxTest]    

    return (xTrain, xTest, yTrain, yTest)






file = "winequality-red.csv"
data = getWineData(file)
xList, labels, names = wineData2ListLabelsNames(data)


nrows = len(xList)
ncols = len(xList[0])

prop = 0.3

xTrain, xTest, yTrain, yTest = wineList2TrainTest(xList,labels,prop)







#train a series of models on random subsets of the training data
#collect the models in a list and check error of composite as list grows

#maximum number of models to generate
numTreesMax = 30

#tree depth - typically at the high end
treeDepth = 5

#initialize a list to hold models
modelList = []
predList = []
eps = 0.1

#initialize residuals to be the labels y
residuals = list(yTrain)

for iTrees in range(numTreesMax):

    modelList.append(DecisionTreeRegressor(max_depth=treeDepth))
    modelList[-1].fit(xTrain, residuals)

    #make prediction with latest model and add to list of predictions
    latestInSamplePrediction = modelList[-1].predict(xTrain)

    #use new predictions to update residuals
    residuals = [residuals[i] - eps * latestInSamplePrediction[i] for i in range(len(residuals))]

    latestOutSamplePrediction = modelList[-1].predict(xTest)
    predList.append(list(latestOutSamplePrediction))


#build cumulative prediction from first "n" models
mse = []
allPredictions = []
for iModels in range(len(modelList)):

    #add the first "iModels" of the predictions and multiply by eps
    prediction = []
    for iPred in range(len(xTest)):
        prediction.append(sum([predList[i][iPred] for i in range(iModels + 1)]) * eps)

    allPredictions.append(prediction)
    errors = [(yTest[i] - prediction[i]) for i in range(len(yTest))]
    mse.append(sum([e * e for e in errors]) / len(yTest))


nModels = [i + 1 for i in range(len(modelList))]

plot.plot(nModels,mse)
plot.axis('tight')
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Mean Squared Error')
plot.ylim((0.0, max(mse)))
plot.show()

print('Minimum MSE')
print(min(mse))

#printed output
#Minimum MSE
#0.405031864814