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
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plot
import csv

def S(z, gamma):
    if gamma >= abs(z):
        return 0.0
    return (z/abs(z))*(abs(z)-gamma)
    

def getWineData(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return (data)
    
    
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


def crossValLARSBeta(xNormalized, labelNormalized, nrows, ncols, nSteps, stepSize):

    nxval = 10
    
    beta = [0.0] * ncols

    errors = []
    for i in range(nSteps):
        b = []
        errors.append(b)
        
        
    for ixval in range(nxval):
        
        idxTest = [a for a in range(nrows) if a%nxval == ixval*nxval]
        idxTrain = [a for a in range(nrows) if a%nxval != ixval*nxval]
        
        xTrain = [xNormalized[r] for r in idxTrain]
        xTest = [xNormalized[r] for r in idxTest]
        
        labelTrain = [labelNormalized[r] for r in idxTrain]
        labelTest = [labelNormalized[r] for r in idxTest]
    
        nrowsTrain = len(idxTrain)
        nrowsTest = len(idxTest)
    
        beta = [0.0] * ncols
         
        
        betaMat = []
        betaMat.append(list(beta))
                
        for iStep in range(nSteps):
    
            residuals = [0.0] * nrows
            
            for j in range(nrowsTrain):
                labelsHat = sum([xTrain[j][k]*beta[k] for k in range(ncols)])
                residuals[j] = labelTrain[j] - labelsHat
                
            corr = [0.0] * ncols
                
            for j in range(ncols):
                corr[j] = sum([xTrain[k][j] * residuals[k] \
                for k in range(nrowsTrain)]) / nrowsTrain
                            
            iStar = 0
            corrStar = corr[0]
                
            for j in range(1,(ncols)):
                if abs(corrStar) < abs(corr[j]):
                    iStar = j
                    corrStar = corr[j]
                
            beta[iStar] += stepSize * corrStar / abs(corrStar)
            betaMat.append(list(beta))
            
            for j in range(nrowsTest):
                labelsHat = sum([xTest[j][k] * beta[k] for k in range(ncols)])
                err = labelTest[j] - labelsHat
                errors[iStep].append(err)
        
    cvCurve = []
    for errVect in errors:
        mse = sum([x*x for x in errVect])/len(errVect)
        cvCurve.append(mse)
        
        
    return (betaMat, cvCurve)



def plotLARSBinaryCoefValues(betaMat, nameList, ncols, nSteps):

    print(nameList)
    
    for i in range(ncols):
        coefCurve = [betaMat[k][i] for k in range(nSteps)]
        xaxis = range(nSteps)
        plot.plot(xaxis,coefCurve)
    
    plot.xlabel("Steps Taken")
    plot.ylabel("Coefficient Values")
    
    
    plot.show()



def calcLARSBinaryLabels(xNormalized, labelNormalized, nrows, ncols, nSteps, stepSize):

    
    beta = [0.0] * ncols
        
    betaMat = []
    betaMat.append(list(beta))
        
    nzList = []
        
    for iStep in range(nSteps):
    
        residuals = [0.0] * nrows
            
        for j in range(nrows):
            
            labelsHat = sum([xNormalized[j][k]*beta[k] for k in range(ncols)])
            residuals[j] = labelNormalized[j] - labelsHat
                
                
        corr = [0.0] * ncols
                
        for j in range(ncols):
            corr[j] = sum([xNormalized[k][j] * residuals[k] \
            for k in range(nrows)]) / nrows
                            
        iStar = 0
        corrStar = corr[0]
                
        for j in range(1,(ncols)):
            if abs(corrStar) < abs(corr[j]):
                iStar = j
                corrStar = corr[j]
                
        beta[iStar] += stepSize * corrStar / abs(corrStar)
        betaMat.append(list(beta))
            
            
        nzBeta = [index for index in range(ncols) if beta[index] != 0.0]
            
        for q in nzBeta:
            if ( q in nzList) == False:
                nzList.append(q)

        names = ['V' + str(i) for i in range(ncols)]
        nameList = [names[nzList[i]] for i in range(len(nzList))]        
       
        
    return (betaMat, nameList)
    



file = 'sonar.all-data'


data = getSonarData(file)

xNum, labels = getSonarDataListLabels(data)
   
xMeans, xSD, nrows, ncols = calcSonarMeansSD(xNum)

xNormalized, labelNormalized = normalizeSonarListLabelData(xNum, labels, xMeans, xSD, nrows, ncols)

nSteps = 350
stepSize = 0.004

betaMat, nameList = calcLARSBinaryLabels(xNormalized, labelNormalized, nrows, ncols, nSteps, stepSize)


plotLARSBinaryCoefValues(betaMat, nameList, ncols, nSteps)




# Now need to implement LARS algorithm for sonar data




#betaMat, nameList = calcGlmNet(xNormalized, labelNormalized, nrows, ncols, nSteps, lamMult)

#plotCoefValues(betaMat)


                



