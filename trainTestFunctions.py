# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:47:35 2015

@author: Craig
"""



import random


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



def wineBagMultiClass(xTrain, xTest, yTrain, yTest, numTreesMax, treeDepth):

    #initialize a list to hold models
    modelList = []
    predList = []
    
    #number of samples to draw for stochastic bagging
    nBagSamples = int(len(xTrain) * 0.5)
    
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
    
    return (nModels, mse)


