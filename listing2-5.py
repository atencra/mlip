# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 00:00:36 2015

@author: craig
"""


__author__ = 'mike_bowles'
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")


# read rocks versus mines data into pandas data frame
rocksVMines = pd.read_csv(target_url, header=None, prefix="V")

print(rocksVMines.head())
print(rocksVMines.tail())


# print summary of data frame
summary = rocksVMines.describe()
print(summary)


