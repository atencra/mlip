# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:02:51 2015

@author: craig
"""

import math

# made up numbers
target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]

error = []

for i in range(len(target)):
    error.append(target[i]-prediction[i])
    
print "Errors "
print error
print "\n"


squaredError = []
absError = []
for val in error:
    squaredError.append(val*val)
    absError.append(abs(val))
    
    
print "Squared Error"
print squaredError    
print "\n"


print "Absolute Value of Error"
print absError
print "\n"


print "MSE = "
print sum(squaredError)/len(squaredError)
print "\n"


print "RMSE = "
print math.sqrt(sum(squaredError)/len(squaredError))
print "\n"


print "MAE = "
print sum(absError)/len(squaredError)
print "\n"




targetDeviation = []
targetMean = sum(target)/len(target)
for val in target:
    targetDeviation.append((val-targetMean)*(val-targetMean))


print "Target Variance = "
print sum(targetDeviation)/len(targetDeviation)
print "\n"



print "Target Standard Deviation = "
print math.sqrt(sum(targetDeviation)/len(targetDeviation))


    
    
    
    
    