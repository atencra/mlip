# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:33:49 2015

@author: Craig
"""

_author_ = 'mike_bowles'
#import urllib2
import sys

#target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
#"databases/undocumented/connectionist-bench/sonar/sonar.all-data")

#data = urllib2.urlopen(target_url)

import csv

    
def getSonarData(file):

    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        
    return (data)


file = "sonar.all-data"   


data = getSonarData(file)

xList = []
labels = []
for line in data:
    row = line[0].strip().split(",")
    xList.append(row)
    
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Cols of Data = " + str(len(xList[1])))

