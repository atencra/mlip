# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:33:49 2015

@author: Craig
"""

_author_ = 'mike_bowles'
import urllib2
import sys
import numpy as np

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")

data = urllib2.urlopen(target_url)

xList = []
labels = []
for line in data:
    row = line.strip().split(",")
    xList.append(row)

nrow = len(xList)
ncol = len(xList[1])

type = [0] * 3
colCounts = []

#generate summary statistics for column 3 (e.g)
col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))

colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)

sys.stdout.write("Mean = " + str(colMean) + '\n' + "Standard Deviation = " + str(colsd) + "\n")

#calculate quantile boundaries
ntiles = 4

percentBdry = []

for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))

sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")

print(percentBdry)
sys.stdout.write(" \n")




#run again with 10 equal intervals
ntiles = 10

percentBdry = []

for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles ))

sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")




# The last column contains categorical variables
col = 60
colData = []
for row in xList:
    colData.append(row[col])
    
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




