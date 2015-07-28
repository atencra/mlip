# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 00:06:20 2015

@author: craig
"""



__author__ = 'mike_bowles'

import pandas as pd
import matplotlib.pyplot as plot

file = "sonar.all-data"


# read rocks versus mines data into pandas data frame
rocksVMines = pd.read_csv(file, header=None, prefix="V")

len(rocksVMines)

for i in range(len(rocksVMines)):
    # assign color based on "M" or "R" labels
    if rocksVMines.iat[i,60] == "M":
        pcolor = "red"
    else:
        pcolor = "blue"
        
    # plot rows of data as if they were series data
    dataRow = rocksVMines.iloc[i,0:60]
    dataRow.plot(color=pcolor)

plot.xlabel("Attribute Index")
plot.ylabel("Attirbute Values")
        
        
        
        
        
        
        
        
        
        