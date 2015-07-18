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
from math import sqrt, cos, log
import matplotlib.pyplot as plot
import csv

    

#--------------------------------------------------------------------------
# Function Definitions
#--------------------------------------------------------------------------

def getAbaloneData(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return (data)


def getAbaloneListLabelsNames(data):

    xList = []
    labels = []

    for line in data:
        labels.append(float( line[-1]  )) # label is last element
        xList.append( line[:-1] ) # don't include last element
        
    names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', \
    'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
            
    return (xList, labels, names)


def codeAbaloneSexAttribute(xList):
    xCoded = []
    for row in xList:
        codedSex = [0.0, 0.0]
        if row[0] == 'M': codedSex[0] = 1.0
        if row[0] == 'F': codedSex[1] = 1.0
            
        numRow = [float(row[i]) for i in range(1,len(row))]
        rowCoded = list(codedSex) + numRow
        xCoded.append(rowCoded)

    namesCoded = ['Sex1', 'Sex2', 'Length', 'Diameter', 'Height', \
        'Whole weight', 'Shucked weight', 'Viscera weight', \
        'Shell weight', 'Rings']
        
    return (xCoded, namesCoded)
    
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
        rowNormalized = [(xList[i][j] - xMeans[j])/xSD[j] \
            for j in range(ncols)]
        xNormalized.append(rowNormalized)
    
    meanLabel = sum(labels)/nrows    
    sdLabel = sqrt(sum([(labels[i] - meanLabel) * (labels[i] - meanLabel) for i in range(nrows)])/nrows)
    
    
    labelNormalized = [(labels[i] - meanLabel) / sdLabel \
        for i in range(nrows)]
    
    return (xNormalized, labelNormalized)


def calcLARS(xNormalized, labelNormalized, nrows, ncols, nSteps, stepSize):

    
    beta = [0.0] * ncols
        
    betaMat = []
    betaMat.append(list(beta))
        
    nzList = []
        
    for iStep in range(nSteps):
    
        residuals = [0.0] * nrows
            
        for j in range(nrows):
            
            labelsHat = sum([xNormalized[j][k] * beta[k] \
                for k in range(ncols)])
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
       
        
    return (betaMat, nzList)



def plotAbaloneCatVar(betaMat, nameList):

    print(nameList)
    
    for i in range(ncols):
        #plot range of beta values for each attribute
        coefCurve = [betaMat[k][i] for k in range(nSteps)]
        xaxis = range(nSteps)
        plot.plot(xaxis, coefCurve)
        
    plot.xlabel("Steps Taken")
    plot.ylabel(("Coefficient Values"))
    plot.show()        
 

#--------------------------------------------------------------------------
# Main Function Calls
#--------------------------------------------------------------------------


file = 'abalone.data'

data = getAbaloneData(file)

xList, labels, names = getAbaloneListLabelsNames(data)

xCoded, namesCoded = codeAbaloneSexAttribute(xList)
    
xMeans, xSD, nrows, ncols = calcMeansSD(xCoded)

xNormalized, labelNormalized = normalizeListLabelData(xCoded, xMeans, xSD, nrows, ncols)

nSteps = 350
stepSize = 0.004
    
betaMat, nzList = calcLARS(xNormalized, labelNormalized, nrows, ncols, nSteps, stepSize)

nameList = [namesCoded[nzList[i]] for i in range(len(nzList))]

plotAbaloneCatVar(betaMat, nameList)


