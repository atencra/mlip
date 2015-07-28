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
from math import sqrt
import matplotlib.pyplot as plot
import csv



import urllib2
import numpy
from sklearn import datasets, linear_model
from sklearn.linear_model import LassoCV
from math import sqrt
import matplotlib.pyplot as plot
    

#--------------------------------------------------------------------------
# Function Definitions
#--------------------------------------------------------------------------

def getWineData(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return (data)


def getWineDataListLabels(data):

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

    
        
def calcXlistMeansSD(xList):
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


def normalizeXlistLabelData(xList, xMeans, xSD, nrows, ncols):
    
    xNormalized = []
    for i in range(nrows):
        rowNormalized = [(xList[i][j] - xMeans[j])/xSD[j] for j in range(ncols)]
        xNormalized.append(rowNormalized)
    
    meanLabel = sum(labels)/nrows    
    sdLabel = sqrt(sum([(labels[i] - meanLabel) * (labels[i] - meanLabel) for i in range(nrows)])/nrows)
    
    
    labelNormalized = [(labels[i] - meanLabel) / sdLabel \
        for i in range(nrows)]
    
    return (xNormalized, labelNormalized)
    


def plotWineLassoCV(wineModel):

    plot.plot(wineModel.alphas_, wineModel.mse_path_, ':')
    plot.plot(wineModel.alphas_, wineModel.mse_path_.mean(axis=-1), \
        label="Average MSE Across Folds", linewidth=2)
    plot.axvline(wineModel.alpha_, linestyle="--",\
        label="CV Estimate of Best Alpha")
        
    plot.semilogx()
    plot.legend()
    ax = plot.gca()
    ax.invert_xaxis()
    plot.xlabel("Alpha")
    plot.ylabel("Mean Squared Error")
    plot.axis("tight")
    plot.show()        
    
    # print alpha that minimizes the CV-error
    print "Alpha that minimizes CV error = %.4f\n" % wineModel.alpha_
    print "Minimum MSE = %.4f\n" % min(wineModel.mse_path_.mean(axis=-1))
 

#--------------------------------------------------------------------------
# Main Function Calls
#--------------------------------------------------------------------------

file = 'winequality-red.csv'

data = getWineData(file)

xList, labels, names = getWineDataListLabels(data)

xMeans, xSD, nrows, ncols = calcXlistMeansSD(xList)

xNormalized, labelNormalized = normalizeXlistLabelData(xList, xMeans, xSD, nrows, ncols)

# Un-normalized labels
Y = numpy.array(labels)

# Normalized labels
Y = numpy.array(labelNormalized)

# Un-normalized X's
X = numpy.array(xList)

# Normalized X's
X = numpy.array(xNormalized)


# Call LassoCV from sklearn.linear_model
wineModel = LassoCV(cv=10).fit(X,Y)

plotWineLassoCV(wineModel)


  







