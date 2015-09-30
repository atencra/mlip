__author__ = 'mike-bowles'

import urllib2
import numpy
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import random
from math import sqrt
import matplotlib.pyplot as plot


import random
import sys
sys.path.insert(0, '\Users\craig\OneDrive\mlip')
import wineUtilityFunctions as wineUF
import trainTestFunctions as ttf




file = 'winequality-red.csv'
data = wineUF.wineCSV2Data(file)
xList, labels, names = wineUF.wineData2ListLabelsNames(data)

nrows = len(xList)
ncols = len(xList[0])



# Get in error when trying to use this function - why? Says it's 
# not in the module????
prop = 0.3 # take 30% of sample for test set
xTrain, xTest, yTrain, yTest = ttf.wineList2TrainTest(xList,labels,prop)






#train a series of models on random subsets of the training data
#collect the models in a list and check error of composite as list grows

#maximum number of models to generate
numTreesMax = 30

#tree depth - typically at the high end
treeDepth = 1



#def wineBagMultiClass(xTrain, xTest, yTrain, yTest, numTreesMax, treeDepth):
#
#    #initialize a list to hold models
#    modelList = []
#    predList = []
#    
#    #number of samples to draw for stochastic bagging
#    nBagSamples = int(len(xTrain) * 0.5)
#    
#    for iTrees in range(numTreesMax):
#        idxBag = []
#        for i in range(nBagSamples):
#            idxBag.append(random.choice(range(len(xTrain))))
#        xTrainBag = [xTrain[i] for i in idxBag]
#        yTrainBag = [yTrain[i] for i in idxBag]
#    
#        modelList.append(DecisionTreeRegressor(max_depth=treeDepth))
#        modelList[-1].fit(xTrainBag, yTrainBag)
#    
#        #make prediction with latest model and add to list of predictions
#        latestPrediction = modelList[-1].predict(xTest)
#        predList.append(list(latestPrediction))
#    
#    
#    #build cumulative prediction from first "n" models
#    mse = []
#    allPredictions = []
#    for iModels in range(len(modelList)):
#    
#        #average first "iModels" of the predictions
#        prediction = []
#        for iPred in range(len(xTest)):
#            prediction.append(sum([predList[i][iPred] for i in range(iModels + 1)])/(iModels + 1))
#    
#        allPredictions.append(prediction)
#        errors = [(yTest[i] - prediction[i]) for i in range(len(yTest))]
#        mse.append(sum([e * e for e in errors]) / len(yTest))
#    
#    
#    nModels = [i + 1 for i in range(len(modelList))]
#    
#    return (nModels, mse)





nModels, mse = ttf.wineBagMultiClass(xTrain, xTest, yTrain, yTest, numTreesMax, treeDepth)

plot.plot(nModels,mse)
plot.axis('tight')
plot.xlabel('Number of Tree Models in Ensemble')
plot.ylabel('Mean Squared Error')
plot.ylim((0.0, max(mse)))
plot.show()

print('Minimum MSE')
print(min(mse))

#with treeDepth = 1
#Minimum MSE
#0.516236026081


#with treeDepth = 5
#Minimum MSE
#0.39815421341

#with treeDepth = 12 & numTreesMax = 100
#Minimum MSE
#0.350749027669