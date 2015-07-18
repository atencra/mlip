# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:14:31 2015

@author: craig
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:57:48 2015

@author: craig
"""


## some_file.py
#import sys
#sys.path.insert(0, '/path/to/application/app/folder')
#
#import file



import urllib2
target_url = ("http://archive.ics.uci.edu/ml/machine-learning-"
"databases/abalone/abalone.data")
#read abalone data
data = urllib2.urlopen(target_url)



xList = []
labels = []
for line in data:
    print line
    #split on semi-colon
    row = line.strip().split(",")
    #put labels in separate array and remove label from row
    labels.append(float(row.pop()))
    #form list of list of attributes (all strings)
    xList.append(row)
names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', \
'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

print labels