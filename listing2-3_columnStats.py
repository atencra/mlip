# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:33:49 2015

@author: Craig
"""

#_author_ = 'mike_bowles'
#import urllib2
import sys
import numpy as np
import csv

    
def readFileSonar(file):

    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        
    return (data)



def colStatsSonar(xList, col, ntiles):
    
    #generate summary statistics for column 3 (e.g)
#    col = 3
    colData = []
    for row in xList:
        colData.append(float(row[col]))
    
    colArray = np.array(colData)
    colMean = np.mean(colArray)
    colsd = np.std(colArray)
    
    sys.stdout.write("Mean = " + str(colMean) + '\n' + "Standard Deviation = " + str(colsd) + "\n")
    
    #calculate quantile boundaries
#    ntiles = 4
    
    percentBdry = []
    
    for i in range(ntiles+1):
        percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
    
    sys.stdout.write("\nBoundaries for " + str(ntiles) + " Equal Percentiles \n")
    
    print(percentBdry)
    sys.stdout.write(" \n")




def countCatVarSonar(xList):

    # The last column contains categorical variables
    colData = []
    for row in xList:
        colData.append(row[-1]) # Get data from last column
        
    unique = set(colData)
    sys.stdout.write("Unique Label Values \n")
    print(unique)
    
    
    #count up number of elements having each value
    
    catDict = dict(zip(list(unique),range(len(unique))))
    
    catCount = [0] * 2
    
    for elt in colData:
        catCount[catDict[elt]] += 1
    
    
    sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
    print(list(unique))
    print(catCount)





file = "sonar.all-data"   

xList = readFileSonar(file)

col = 3
ntiles = 4
colStatsSonar(xList, col, ntiles)

col = 3
ntiles = 10
colStatsSonar(xList, col, ntiles)

countCatVarSonar(xList)






