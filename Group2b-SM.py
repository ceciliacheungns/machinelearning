# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 23:15:35 2017

@author: siowmeng
"""

import pandas as pd
import numpy as np
from sklearn import neighbors, cross_validation, preprocessing

def compMeanStd(dataMatrix):
    # Use sample standard deviation (denominator = n - 1)
    return np.mean(dataMatrix, axis = 0), np.std(dataMatrix, axis = 0, ddof = 1)

def zScore(dataMatrix, meanValue, stdValue):
    return ((dataMatrix - meanValue) / stdValue)

np.random.seed(2017)

redWines = pd.read_csv('winequality-red.csv', sep = ';')
redWines['GoodWine'] = (redWines['quality'] >= 6)

# Exclude 'quality' and 'GoodWine' columns for training
redWinesX = redWines.iloc[ : , :-2].as_matrix()
# sci-kit learn preprocessing scale function uses population std dev
#redWines.X.scaled = preprocessing.scale(redWines.X)

# Use 'GoodWine' as outcome variable
redWinesY = redWines['GoodWine'].as_matrix()

# Shuffle and split the data into two sets (training = to be used for K-fold cross validation)
rs = cross_validation.ShuffleSplit(len(redWines), n_iter = 1, test_size = 0.5, random_state = 2017)
for trainIndex, testIndex in rs:
    inTrain = trainIndex
    inTest = testIndex

trainX = redWinesX[inTrain]
trainY = redWinesY[inTrain]

testX = redWinesX[inTest]
testY = redWinesY[inTest]

meanVal, stdVal = compMeanStd(trainX)
# Scale data using only the estimates from (training + validation) data
trainXscaled = zScore(trainX, meanVal, stdVal)
testXscaled = zScore(testX, meanVal, stdVal)

# For the training set, use k-fold cross validation to split into training and validation
kf = cross_validation.KFold(len(trainXscaled), n_folds = 5, random_state = 2017)
for k in np.arange(1, 502, 5):
    acc = []
    sens = []
    spec = []
    for trainIndex, valIndex in kf:
        clf = neighbors.KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
        clf.fit(trainXscaled[trainIndex], trainY[trainIndex]) # Fit on training data
        # Predict on validation data
        prediction = clf.predict(trainXscaled[valIndex]) # Predict on validation data
        actual = trainY[valIndex] # Actual outcome values of validation data
        numTP = sum((actual == True) & (prediction == True))
        numTN = sum((actual == False) & (prediction == False))
        acc.append((numTP + numTN) / len(actual))
        sens.append(numTP / sum(actual == True))
        spec.append(numTN / sum(actual == False))
    if k == 1:
        perfK = [[k, np.mean(acc), np.mean(sens), np.mean(spec)]]
    else:
        perfK = np.append(perfK, [[k, np.mean(acc), np.mean(sens), np.mean(spec)]], axis = 0)

bestK = perfK[np.argmax(perfK, axis = 0)[1], 0] # Select bestK based on accuracy
clf = neighbors.KNeighborsClassifier(n_neighbors = int(bestK), weights = 'uniform')
# Test Phase: Re-train on all training data (all training + validation data used in K-fold)
clf.fit(trainXscaled, trainY)
prediction = clf.predict(testXscaled) # Predict on test data
actual = testY # Actual outcome values of test data
numTP = sum((actual == True) & (prediction == True))
numTN = sum((actual == False) & (prediction == False))
numFN = sum((actual == True) & (prediction == False))
numFP = sum((actual == False) & (prediction == True))
acc = (numTP + numTN) / len(actual)
sens = numTP / sum(actual == True)
spec = numTN / sum(actual == False)
# Confusion matrix - Generalisation Error
testConfMat = pd.DataFrame([[numTP, numFN], [numFP, numTN]], 
                           columns = ['True', 'False'], 
                           index = ['True', 'False'])

# Finally, train on all data for future use
meanAll, stdAll = compMeanStd(redWinesX)
redWinesXscaled = zScore(redWinesX, meanAll, stdAll) # Scale using estimates from all data

clf.fit(redWinesXscaled, redWinesY) # Fit on all data
