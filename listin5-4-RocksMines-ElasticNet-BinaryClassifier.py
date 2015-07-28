# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:57:48 2015

@author: craig
"""


## some_file.py
#import sys
#sys.path.insert(0, '/path/to/application/app/folder')
#
#import file



#import urllib2
from math import sqrt, fabs, exp
import matplotlib.pyplot as plot
from sklearn.linear_model import enet_path
from sklearn.metrics import roc_auc_score, roc_curve
import numpy
import csv


#----------- Bug in his code when using elastic net -------------
   
    
def getSonarData(file):

    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        
    return (data)



def getSonarDataListLabels(data):

    xNum = []
    labels = []
    
    for row in data:
        lastCol = row.pop()

        if lastCol == "M":
            labels.append(1.0)
        else:
            labels.append(0.0)
        attrRow = [float(elt) for elt in row]
        
        xNum.append(attrRow)
        
    return (xNum, labels)


    
    
def calcSonarMeansSD(xNum):
    nrows = len(xNum)
    ncols = len(xNum[1])
    
    xMeans = []
    xSD = []
    for i in range(ncols):
        col = [xNum[j][i] for j in range(nrows)]
        mean = sum(col) / nrows
        xMeans.append(mean)
        colDiff = [(xNum[j][i] - mean) for j in range(nrows)]
        sumSq = sum([colDiff[i]*colDiff[i] for i in range(nrows)])
        stdDev = sqrt(sumSq/nrows)
        xSD.append(stdDev)
    return (xMeans, xSD, nrows, ncols)


def normalizeSonarListLabelData(xNum, labels, xMeans, xSD, nrows, ncols):
    
    xNormalized = []
    
    for i in range(nrows):
        
        rowNormalized = [(xNum[i][j] - xMeans[j])/xSD[j] \
            for j in range(ncols)]
        xNormalized.append(rowNormalized)
    
    meanLabel = sum(labels)/nrows    
    
    sdLabel = sqrt(sum([(labels[i] - meanLabel) * (labels[i] - meanLabel) \
        for i in range(nrows)])/nrows)

    labelNormalized = [(labels[i] - meanLabel) / sdLabel \
        for i in range(nrows)]
    
    return (xNormalized, labelNormalized)


def crossValENetRocksMines(xNormalized, labelNormalized, nrows, ncols):

    nxval = 10

        
    for ixval in range(nxval):
        
        idxTest = [a for a in range(nrows) if a%nxval == ixval*nxval]
        idxTrain = [a for a in range(nrows) if a%nxval != ixval*nxval]
        
        xTrain = [xNormalized[r] for r in idxTrain]
        xTest = [xNormalized[r] for r in idxTest]
        
        labelTrain = [labelNormalized[r] for r in idxTrain]
        labelTest = [labelNormalized[r] for r in idxTest]
        
        alphas, coefs,_ = enet_path(xTrain, labelTrain, l1_ratio = 0.8, \
        fit_intercept=False, return_models=False)
        
        if ixval == 0:
            pred = numpy.dot(xTest, coefs)
            yOut = labelTest
        else:
            yTemp = numpy.array(yOut)
            yOut = numpy.concatenate((yTemp,labelTest),axis=0)
            
            predTemp = numpy.array(pred)
            pred = numpy.concatenate((predTemp,numpy.dot(xTest,coefs)),axis=0)

    misClassRate = []
    _,nPred = pred.shape
    for iPred in range(1,nPred):
        predList = list(pred[:,iPred])
        errCnt = 0.0
        
        for irow in range(nrows):
            if (predList[irow] < 0.0) and (yOut[irow]>=0.0):
                errCnt += 1.0
            elif (predList[irow] >= 0.0) and (yOut[irow] < 0.0):
                errCnt += 1.0
        misClassRate.append(errCnt/nrows)
        
    return (misClassRate, alphas)
        

def plotMisClassRateRocksMines(misClassRate, alphas):
    minError = min(misClassRate)
    idxMin = misClassRate.index(minError)
    
    plotAlphas = list(alphas[1:len(alphas)])
    plot.figure()
    plot.plot(plotAlphas, misClassRate, \
    label = "Misclassification Error Across Folds", linewidth=2)
    plot.axvline(plotAlphas[idxMin], linestyle="--",\
    label = "CV Estimate of Best Alpha")
    
    plot.legend()
    plot.semilogx()
    ax = plot.gca()
    ax.invert_xaxis()
    plot.xlabel("Alpha")
    plot.ylabel("Misclassification Error")
    plot.axis("tight")
    plot.show()






file = 'sonar.all-data'

data = getSonarData(file)

xNum, labels = getSonarDataListLabels(data)
   
xMeans, xSD, nrows, ncols = calcSonarMeansSD(xNum)

xNormalized, labelNormalized = normalizeSonarListLabelData(xNum, labels, xMeans, xSD, nrows, ncols)

misClassRate, alphas = crossValENetRocksMines(xNormalized, labelNormalized, nrows, ncols)

plotMisClassRateRocksMines(misClassRate, alphas)




