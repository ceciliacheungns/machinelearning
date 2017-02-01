#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 20:18:48 2017

@author: cheungcecilia
"""

#Machine Learning - Group exercise 1


import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

data = pd.read_csv('winequality-red.csv', delimiter = ';')

#classify 'good wine' - 0 or 1

data['good wine'] = 0

for i in range(0, len(data.index)):
    #print(i)
    if data.iloc[i, 11] >= 6:
        data.iloc[i, 12] = 1
    else:
        data.iloc[i, 12] = 0

#shuffle data
  
def shuffle(df):
    datashuffle = df.reindex(np.random.permutation(df.index))
    return datashuffle

data_sh = shuffle(data)

train = data_sh.iloc[range(0, 800), ]

test = data_sh.iloc[range(800, 1599), ]

#z-score transform
  
cols = list(train.columns)
    
for col in cols:
    col_zscore = col + '_zscore'
    train[col_zscore] = (train[col] - train[col].mean())/train[col].std(ddof=0)

del train['good wine_zscore']
        
cols1 = list(test.columns)

for col in cols1:
    col_zscore = col + '_zscore'
    test[col_zscore] = (test[col] - test[col].mean())/test[col].std(ddof=0)

del test['good wine_zscore']
    
#create list of k values

z = []
a = 1
z.append(1)

for i in range(1, 500):
    if a < 500:
        b = a + 5
        z.append(b)
        a = b
    else:
        exit 
z.pop()
    
kclassifier = list(z)


#evaluate classifier using 5-fold cross validation

total = train.append(test)
total_x = total.iloc[:,13:24]
total_y = data_sh.iloc[:,12]

total_x_train = total_x.iloc[0:799,]
total_y_train = total_y.iloc[0:799,]

#5-fold cross validation 

knn = KNeighborsClassifier(n_neighbors = i)
    
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(knn, total_x_train, total_y_train, cv=5, scoring='accuracy')
print(scores)
print(scores.mean()) #mean score from 5 fold cross validation

print(1-scores.mean()) 

# search for an optimal value of K for KNN

k_range = z # range of k we want to try
k_scores = [] # empty list to store scores

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, total_x_train, total_y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

print(k_scores) #mean score from k-nearest neighbours

#classification error

k_scores_inv = []
for i in range(0, len(k_scores)):
    print(i)
    k_scores_inverse = 1 - k_scores[i]
    k_scores_inv.append(k_scores_inverse)

#create dataframe showing results 

df_k_scores_inv = pd.DataFrame(
    {'Classification Error': k_scores_inv,
     'k-value': z
    })

print(df_k_scores_inv)

#optimal k value is 66

#test using k = 66 on test data set

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

total_x_test = test.iloc[:,13:24]
total_y_test = test.iloc[:,12]

knn = KNeighborsClassifier(n_neighbors = 96)
knn.fit(total_x_train, total_y_train)
predictions = knn.predict(total_x_test)
accuracy = accuracy_score(total_y_test, predictions)
    
print(accuracy)

#confusion matrix

matrix = confusion_matrix(total_y_test, predictions)
print(matrix)






