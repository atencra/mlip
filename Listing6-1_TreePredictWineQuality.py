# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:05:28 2015

@author: craig
"""

__author__ = 'mike-bowles'
from sklearn.tree import DecisionTreeRegressor
import sys
sys.path.insert(0, '\Users\craig\OneDrive\mlip')
import wineUtilityFunctions as wineUF



file = 'winequality-red.csv'

data = wineUF.wineCSV2Data(file)

xList, labels, names = wineUF.wineData2ListLabelsNames(data)


wineTree = DecisionTreeRegressor(max_depth=3)

wineTree.fit(xList, labels)

#with open("wineTree.dot", 'w') as f:
#
#f = tree.export_graphviz(wineTree, out_file=f)


#Note: The code above exports the trained tree info to a
#Graphviz "dot" file.
#Drawing the graph requires installing GraphViz and the running the
#following on the command line
#dot -Tpng wineTree.dot -o wineTree.png
# In Windows, you can also open the .dot file in the GraphViz
#gui (GVedit.exe)]