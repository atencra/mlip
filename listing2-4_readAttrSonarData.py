# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 23:50:06 2015

@author: craig
"""

_author_ = 'mike bowles'
import pylab
import scipy.stats as stats
import csv


    
def readFileSonar(file):

    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        
    return (data)


def getListFromDataSonar(data):
    
    # arrange data into list for labels and list of lists for attributes
    xList = []
    labels = []
    
    for line in data:
        # split on comma
#        row = line.strip().split(",")
        labels.append(line.pop())
        xList.append(line)

    return (xList, labels)


file = "sonar.all-data"
data = readFileSonar(file)
xList, labels = getListFromDataSonar(data)



nrow = len(xList)
ncol = len(xList[1])

type = [0] * 3
colCounts = []

# generate summary statistics for column 3 (e.g.)
col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))




stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()






