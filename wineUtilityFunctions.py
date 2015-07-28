# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:51:47 2015

@author: Craig
"""

#--------------------------------------------------------------------------
# Wine Data Utility Functions
#--------------------------------------------------------------------------


import csv



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
    
    
    