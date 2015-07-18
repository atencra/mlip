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
    print nrows
    print ncols
    
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
    return(xMeans, xSD, nrows, ncols)


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


def calcLARSBeta(xNormalized, labelNormalized, nrows, ncols, nSteps, stepSize):

    beta = [0.0] * ncols

    betaMat = []
    betaMat.append(list(beta))

    for i in range(nSteps):

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
        
    return (betaMat)




file = 'winequality-red.csv'

import csv
with open(file, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)


xList, labels, names = getDataListLabels(data)
    
xMeans, xSD, nrows, ncols = calcMeansSD(xList)

xNormalized, labelNormalized = normalizeListLabelData(xList, xMeans, xSD, nrows, ncols)


nSteps = 350
stepSize = 0.005

betaMat = calcLARSBeta(xNormalized, labelNormalized, nrows, ncols, nSteps, stepSize)
   
     
for i in range(ncols):
    coefCurve = [betaMat[k][i] for k in range(nSteps)]
    xaxis = range(nSteps)
    plot.plot(xaxis,coefCurve)
    
plot.xlabel("Steps Taken")
plot.ylabel("Coefficient Values")
plot.show()
                



