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


def getDataListLabels(data):

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
    
    
def calcMeansSD(xList):
    nrows = len(xList)
    ncols = len(xList[0])
    
    xMeans = []
    xSD = []
    for i in range(ncols):
        col = [xList[j][i] for j in range(nrows)]
        mean = sum(col) / nrows
        xMeans.append(mean)
        colDiff = [(xList[j][i] - mean) for j in range(nrows)]
        sumSq = sum([colDiff[i]*colDiff[i] for i in range(nrows)])
        stdDev = sqrt(sumSq/nrows)
        xSD.append(stdDev)
    return (xMeans, xSD, nrows, ncols)


def normalizeListLabelData(xList, xMeans, xSD, nrows, ncols):
    
    xNormalized = []
    for i in range(nrows):
        rowNormalized = [(xList[i][j] - xMeans[j])/xSD[j] for j in range(ncols)]
        xNormalized.append(rowNormalized)
    
    meanLabel = sum(labels)/nrows    
    sdLabel = sqrt(sum([(labels[i] - meanLabel) * (labels[i] - meanLabel) for i in range(nrows)])/nrows)
    
    
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







def calcGlmNet(xNormalized, labelNormalized, nrows, ncols, nSteps, lamMult):

    alpha = 1.0
    
    xy = [0.0] * ncols
    
    for i in range(nrows):
        for j in range(ncols):
            xy[j] += xNormalized[i][j] * labelNormalized[i]
            
    maxXY = 0.0
    for i in range(ncols):
        val = abs(xy[i])/nrows
        if val > maxXY:
            maxXY = val
            
    lam = maxXY / alpha
    
    beta = [0.0] * ncols
    
    betaMat = []
    betaMat.append(list(beta))
    
    nzList = []
    
    for iStep in range(nSteps):
        
        lam = lam * lamMult
        
        deltaBeta = 100.0
        eps = 0.01
        iterStep = 0
        betaInner = list(beta)
        
        while deltaBeta > eps:
            iterStep += 1
            if iterStep > 100: break
                
            betaStart = list(betaInner)
            
            for iCol in range(ncols):
                
                xyj = 0.0
                for i in range(nrows):
                    labelHat = sum([xNormalized[i][k]*betaInner[k]
                       for k in range(ncols)])
                    residual = labelNormalized[i] - labelHat
                    
                    xyj += xNormalized[i][iCol] * residual
                    
                uncBeta = xyj / nrows + betaInner[iCol]
                betaInner[iCol] = S(uncBeta, lam*alpha) / (1+lam*(1-alpha))

            sumDiff = sum([abs(betaInner[n] - betaStart[n]) for n in range(ncols)])                    

            sumBeta = sum([abs(betaInner[n]) for n in range(ncols)])
            
            deltaBeta = sumDiff / sumBeta
            
        print "iStep = %.0f, iterStep = %.0f\n" % (iStep, iterStep)
        
        beta = betaInner
        
        betaMat.append(beta)
        
        nzBeta = [index for index in range(ncols) if beta[index] != 0.0]
        
        for q in nzBeta:
            if ( q in nzList) == False:
                nzList.append(q)
                
    nameList = [names[nzList[i]] for i in range(len(nzList))]
    print nameList

        
    return (betaMat, nameList)




def plotCoefValues(betaMat):

    nPts = len(betaMat)
    
    for i in range(ncols):
        coefCurve = [betaMat[k][i] for k in range(nPts)]
        xaxis = range(nPts)
        plot.plot(xaxis,coefCurve)
        
    plot.xlabel("Steps Taken")
    plot.ylabel("Coefficient Values")
    
    
    plot.show()
    



file = 'winequality-red.csv'

data = getWineData(file)

xList, labels, names = getDataListLabels(data)
    
xMeans, xSD, nrows, ncols = calcMeansSD(xList)

xNormalized, labelNormalized = normalizeListLabelData(xList, xMeans, xSD, nrows, ncols)


nSteps = 100
lamMult = 0.93

betaMat, nameList = calcGlmNet(xNormalized, labelNormalized, nrows, ncols, nSteps, lamMult)

plotCoefValues(betaMat)


                



