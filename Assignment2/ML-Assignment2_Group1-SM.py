# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 02:52:11 2017

@author: siowmeng
"""

import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
import timeit

#import os
#os.chdir('D:/Imperial MSc/Core Modules/Machine Learning/Assignments/Assignment 2')

seedValue = 2017

# 1. Load to pandas data frame with two columns
spam = pd.read_table('./smsspamcollection/SMSSpamCollection', header = None, names = ['type', 'text'])

# 2. Preprocessing - remove punctuations and digits, change to lowercase
translator = str.maketrans(string.ascii_uppercase, 
                           string.ascii_lowercase, 
                           string.punctuation + string.digits)
spam['text'] = spam['text'].str.translate(translator)

# 3. Shuffle and split: Training (2500 messages) + Validation (1000 messages) + Test (remaining)
train, others = train_test_split(spam, train_size = 2500, random_state = seedValue)
validation, test = train_test_split(others, train_size = 1000, random_state = seedValue)

# 4. Naive Bayes classifier
class NaiveBayesForSpam:
    
    # This function calculates the probability of occurrence of each word in ham/spam messages
    def train(self, hamMessages, spamMessages):
        self.words = set(' '.join(hamMessages + spamMessages).split())
        self.priors = np.zeros(2)
        self.priors[0] = float(len(hamMessages)) / (len(hamMessages) + len(spamMessages))
        self.priors[1] = 1.0 - self.priors[0]
        self.likelihoods = []
        for i, w in enumerate(self.words):
            # Adjust every word by adding 1.0, Laplace estimator
            prob1 = (1.0 + len([m for m in hamMessages if w in m])) / len(hamMessages)
            prob2 = (1.0 + len([m for m in spamMessages if w in m])) / len(spamMessages)
            # Adjust every word to maximally 0.95 probability of occurrence
            self.likelihoods.append([min(prob1, 0.95), min(prob2, 0.95)])
        self.likelihoods = np.array(self.likelihoods).T

    # This function only calculates the probability of occurrences of spam keywords
    # Spam keywords appear in spamMessages more than 20 times more than hamMessages
    def train2(self, hamMessages, spamMessages):
        self.words = set(' '.join(hamMessages + spamMessages).split())
        self.priors = np.zeros(2)
        self.priors[0] = float(len(hamMessages)) / (len(hamMessages) + len(spamMessages))
        self.priors[1] = 1.0 - self.priors[0]
        self.likelihoods = []
        spamkeywords = []
        for i, w in enumerate(self.words):
            prob1 = (1.0 + len([m for m in hamMessages if w in m])) / len(hamMessages)
            prob2 = (1.0 + len([m for m in spamMessages if w in m])) / len(spamMessages)
            if prob1 * 20 < prob2:
                self.likelihoods.append([min(prob1, 0.95), min(prob2, 0.95)])
                spamkeywords.append(w)
        self.words = spamkeywords
        self.likelihoods = np.array(self.likelihoods).T
    
    # This function predicts the category (ham or spam) and the posterior probability
    def predict(self, message):
        posteriors = np.copy(self.priors)
        for i, w in enumerate(self.words):
            # Assume class-conditional independence and multiply each conditional probabilities
            if w in message.lower(): # convert to lower-case
                posteriors *= self.likelihoods[:, i]
            else:
                posteriors *= np.ones(2) - self.likelihoods[:, i]
            posteriors = posteriors / np.linalg.norm(posteriors, ord = 1) # normalise
        if posteriors[0] > 0.5:
            return ['ham', posteriors[0]]
        return ['spam', posteriors[1]]
    
    # This function returns the accuracy and confusion matrix
    def score(self, messages, labels):
        confusion = np.zeros(4).reshape(2, 2)
        for m, l in zip(messages, labels):
            if self.predict(m)[0] == 'ham' and l == 'ham':
                confusion[0, 0] += 1
            elif self.predict(m)[0] == 'ham' and l == 'spam':
                confusion[0, 1] += 1
            elif self.predict(m)[0] == 'spam' and l == 'ham':
                confusion[1, 0] += 1
            elif self.predict(m)[0] == 'spam' and l == 'spam':
                confusion[1, 1] += 1
        return (confusion[0, 0] + confusion[1, 1]) / float(confusion.sum()), confusion

# 5. Explain the code: Difference and Bayes Theorem

# 6. Train using 'train' and 'train2'
trainHam = list(train.loc[train['type'] == 'ham', 'text'])
trainSpam = list(train.loc[train['type'] == 'spam', 'text'])

clf1 = NaiveBayesForSpam()
clf1.train(trainHam, trainSpam)

clf2 = NaiveBayesForSpam()
clf2.train2(trainHam, trainSpam)

# 7. Validation Performance
validationLbls = validation['type'].as_matrix()
validationMsgs = validation['text'].as_matrix()

# Do separate for loop to record the time difference
predictions1 = []
start_time = timeit.default_timer()
#for message in validationMsgs:
#    predictions1.append(clf1.predict(message))
valAcc1, valConf1 = clf1.score(validationMsgs, validationLbls)
elapsed = timeit.default_timer() - start_time
print('Total time to train 1st Naive Bayes classifier:', elapsed, 'seconds')
#predictions1 = np.array(predictions1)
    
predictions2 = []
start_time = timeit.default_timer()
#for message in validationMsgs:
#    predictions2.append(clf2.predict(message))
valAcc2, valConf2 = clf2.score(validationMsgs, validationLbls)
elapsed = timeit.default_timer() - start_time
print('Total time to train 2nd Naive Bayes classifier:', elapsed, 'seconds')
#predictions2 = np.array(predictions2)

# 8. Why is train2 faster? Accuracy on training and validation set

# train2 is faster since it has much less keywords to calculate the posterior probabilities
len(clf1.words)
len(clf2.words)

trainingLbls = train['type'].as_matrix()
trainingMsgs = train['text'].as_matrix()

trainAcc1, trainConf1 = clf1.score(trainingMsgs, trainingLbls)
trainAcc2, trainConf2 = clf2.score(trainingMsgs, trainingLbls)

# Accuracy
print('Accuracy of 1st Naive Bayes classifier (on training set):', trainAcc1)
print('Accuracy of 2nd Naive Bayes classifier (on training set):', trainAcc2)
print('Accuracy of 1st Naive Bayes classifier (on validation set):', valAcc1)
print('Accuracy of 2nd Naive Bayes classifier (on validation set):', valAcc2)

# Why is it more accurate?
# By reducing the number of keywords to look out for in spam message, 
# we are becoming less strict in looking out for spam message.
# As a results, more messages will be classified as ham. This drives up the
# numbers of True Negative (Ham message correctly specified) and
# False Negative (Spam message misclassified as Ham). Since there is much more
# Ham message than Spam message in the dataset, more Ham messages becomes correctly
# classified than Spam messages being misclassified. As a result, the accuracy
# becomes better

# 9. Number of false positives and tweak to algorithm
print('Number of False Positives for 1st classifier:', valConf1[1, 0])
print('Number of False Positives for 2nd classifier:', valConf2[1, 0])

# train2 is exactly doing that. It reduces the False Positives (Ham message misclassified 
# as Spam) at the expense of having more False Negatives (Spam message misclassified as
# Ham). Since there are much more Ham than Spam, this reduction far outweighs the increase
# of False Negatives, improving accuracy as a result.

# 10. train2 test performance

# Retrain on both training and validation data
trainValHam = trainHam + list(validation.loc[validation['type'] == 'ham', 'text'])
trainValSpam = trainSpam + list(validation.loc[validation['type'] == 'spam', 'text'])
clf2.train2(trainValHam, trainValSpam)

testLbls = test['type'].as_matrix()
testMsgs = test['text'].as_matrix()
testAcc2, testConf2 = clf2.score(testMsgs, testLbls)
print('Accuracy of 2nd Naive Bayes classifier (on test set):', testAcc2)
print('Confusion Matrix of 2nd Naive Bayes classifier (on test set):')
print(testConf2)

