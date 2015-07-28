# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 00:00:36 2015

@author: craig
"""


__author__ = 'mike_bowles'
import pandas as pd

file = "sonar.all-data"

# read rocks versus mines data into pandas data frame
rocksVMines = pd.read_csv(file, header=None, prefix="V")

print(rocksVMines.head())
print(rocksVMines.tail())


# print summary of data frame
summary = rocksVMines.describe()
print(summary)


