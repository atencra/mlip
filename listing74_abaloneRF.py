__author__ = 'mike_bowles'

import matplotlib.pyplot as plot
import numpy
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import csv


def getAbaloneData(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return (data)


def getAbaloneListLabelsNames(data):

    xList = []
    labels = []

    for line in data:
        labels.append(float( line[-1]  )) # label is last element
        xList.append( line[:-1] ) # don't include last element
        
    names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', \
    'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
            
    return (xList, labels, names)


def codeAbaloneSexAttribute(xList):
    xCoded = []
    for row in xList:
        codedSex = [0.0, 0.0]
        if row[0] == 'M': codedSex[0] = 1.0
        if row[0] == 'F': codedSex[1] = 1.0
            
        numRow = [float(row[i]) for i in range(1,len(row))]
        rowCoded = list(codedSex) + numRow
        xCoded.append(rowCoded)

    namesCoded = ['Sex1', 'Sex2', 'Length', 'Diameter', 'Height', \
        'Whole weight', 'Shucked weight', 'Viscera weight', \
        'Shell weight', 'Rings']
        
    return (xCoded, namesCoded)


file = 'abalone.data'

data = getAbaloneData(file)

print data[0]

xList, labels, names = getAbaloneListLabelsNames(data)

xCoded, abaloneNames = codeAbaloneSexAttribute(xList)



print abaloneNames


#number of rows and columns in x matrix
nrows = len(xCoded)
ncols = len(xCoded[1])

#form x and y into numpy arrays and make up column names
X = numpy.array(xCoded)
y = numpy.array(labels)

#break into training and test sets.
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30, random_state=531)

#train random forest at a range of ensemble sizes in order to see how the mse changes
mseOos = []
nTreeList = range(50, 500, 10)
for iTrees in nTreeList:
    depth = None
    maxFeat  = 4 #try tweaking
    abaloneRFModel = ensemble.RandomForestRegressor(n_estimators=iTrees, max_depth=depth, max_features=maxFeat,
                                                 oob_score=False, random_state=531)

    abaloneRFModel.fit(xTrain,yTrain)

    #Accumulate mse on test set
    prediction = abaloneRFModel.predict(xTest)
    mseOos.append(mean_squared_error(yTest, prediction))


print("MSE" )
print(mseOos[-1])


#plot training and test errors vs number of trees in ensemble
plot.plot(nTreeList, mseOos)
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Mean Squared Error')
#plot.ylim([0.0, 1.1*max(mseOob)])
plot.show()

# Plot feature importance
featureImportance = abaloneRFModel.feature_importances_

# normalize by max importance
featureImportance = featureImportance / featureImportance.max()
sortedIdx = numpy.argsort(featureImportance)
barPos = numpy.arange(sortedIdx.shape[0]) + .5

plot.barh(barPos, featureImportance[sortedIdx], align='center')

plot.yticks(barPos, abaloneNames[sortedIdx])


plot.xlabel('Variable Importance')
plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plot.show()

# Printed Output:
# MSE
# 4.30971555911