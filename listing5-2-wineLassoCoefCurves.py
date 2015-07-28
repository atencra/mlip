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
    


def plotWineLassoAlphasCoefs(alphas, coefs, names):

    plot.plot(alphas, coefs.T)
    plot.xlabel("Alpha")
    plot.ylabel("Coefs")
    plot.axis("tight")
    plot.semilogx()
    ax = plot.gca()
    ax.invert_xaxis()
    plot.show()        
    
    nattr, nalpha = coefs.shape
    
    nzList = []
    for iAlpha in range(1,nalpha):
        coefList = list(coefs[:,iAlpha])
        nzCoef = [index for index in range(nattr) if coefList[index] != 0.0]
        for q in nzCoef:
            if not(q in nzList):
                nzList.append(q)
        
    nameList = [names[nzList[i]] for i in range(len(nzList))]
    
    print "Attr by model entrance\n" 
    print nameList
    
    # Hard code best attribute from CV as determined from Listing 5-1
    alphaStar = 0.0135613877
    indexLTalphaStar = [index for index in range(100) if alphas[index] > alphaStar]
    
    # index for best alpha
    indexStar = max(indexLTalphaStar)
    
    # coefs corresponding to best alpha
    coefStar = list(coefs[:,indexStar])
    print "Best coefficient values:\n"
    print coefStar
    
 

#--------------------------------------------------------------------------
# Main Function Calls
#--------------------------------------------------------------------------

file = 'winequality-red.csv'

data = getWineData(file)

xList, labels, names = getWineDataListLabels(data)

xMeans, xSD, nrows, ncols = calcXlistMeansSD(xList)

xNormalized, labelNormalized = normalizeXlistLabelData(xList, xMeans, xSD, nrows, ncols)

# Normalized labels
Y = numpy.array(labelNormalized)

# Normalized X's
X = numpy.array(xNormalized)

alphas, coefs, _ = linear_model.lasso_path(X,Y,return_models=False)

plotWineLassoAlphasCoefs(alphas, coefs, names)

  







