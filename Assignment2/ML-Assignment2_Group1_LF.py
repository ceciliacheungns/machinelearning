# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:38:18 2017

@author: louisefallon
"""
#%%libraries

import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split

#%% set seed
seedValue = 2017

#%% load data
spam = pd.read_table('./smsspamcollection/SMSSpamCollection', header = None, names = ['type', 'text'])

#%%
##Pre-process the SMS messages:
##Remove all punctuation and numbers from the SMS messages
spam['cleantext'] = spam['text'].apply(lambda x:''.join([i for i in x 
                                                  if i not in string.punctuation]))                                       
##and change all messages to lower case.
spam['cleantext'] = spam['cleantext'].apply(lambda x: x.lower())                                                    
#%%
train, others = train_test_split(spam, train_size = 2500, random_state = seedValue)
validation, test = train_test_split(others, train_size = 1000, random_state = seedValue)
#%%

class NaiveBayesForSpam:

#calculate conditional probabilities of all words in all messages
    def train (self, hamMessages, spamMessages):
        self.words = set(" ".join (hamMessages + spamMessages).split()) #Create a set of all words used in all texts
        self.priors = np.zeros(2) ##set prior probability to 0.
        self.priors[0] = float(len(hamMessages)) / (len(hamMessages) + len(spamMessages)) 
        self.priors[1] = 1.0 - self.priors[0] 
        self.likelihoods = []
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len([m for m in hamMessages if w in m])) / len(hamMessages)
            prob2 = (1.0 + len([m for m in spamMessages if w in m])) / len(spamMessages) 
            self.likelihoods.append([min(prob1,0.95),min(prob2,0.95)]) #Lagrange
        self.likelihoods = np.array(self.likelihoods).T

#calculate conditional probabilities P(w|ham) and P(w|spam) of all words in all messages
#but only return those that are overrepresented (20x) in spam
    def train2 (self, hamMessages, spamMessages):
        self.words = set(" ".join (hamMessages + spamMessages).split())
        self.priors = np.zeros(2)
        self.priors[0] = float(len(hamMessages)) / (len(hamMessages) + len(spamMessages)) 
        self.priors[1] = 1 - self.priors[0]
        self.likelihoods = []
        spamkeywords = []
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len([m for m in hamMessages if w in m])) / len(hamMessages)
            prob2 = (1.0 + len([m for m in spamMessages if w in m])) / len(spamMessages) 
            if prob1 * 20 < prob2: 
                self.likelihoods.append([min(prob1,0.95),min(prob2,0.95)])
                spamkeywords.append(w)
        self.words = spamkeywords
        self.likelihoods = np.array(self.likelihoods).T

#predict whether a value is spam or ham based on naive bayes
    def predict (self, message):
        posteriors = np.copy (self.priors)
        for i, w in enumerate (self.words):
            if w in message.lower():
                posteriors *= self.likelihoods[:,i]
            else:
                posteriors *= np.ones(2) - self.likelihoods[:,i]
            posteriors = posteriors / np.linalg.norm(posteriors, ord = 1) #Normalising
        if posteriors[0] > 0.5:
            return ['ham', posteriors[0]]
        return ['spam', posteriors[1]]

#produces a confusion matrix with:
#                actual ham     actual spam
# predict ham         #              #
# predict spam        #              #

    def score (self,messages,labels):
        confusion = np.zeros(4).reshape(2,2)
        for m, l in zip(messages, labels):
            if self.predict(m)[0] == 'ham' and l == 'ham':
                confusion[0,0] += 1
            elif self.predict(m)[0] == 'ham' and l == 'spam':
                confusion[0,1] += 1
            elif self.predict(m)[0] == 'spam' and l == 'ham':
                confusion[1,0] += 1
            elif self.predict(m)[0] == 'spam' and l == 'spam':
                confusion[1,1] += 1
        return (confusion[0,0] + confusion[1,1]) / float (confusion.sum()), confusion
        
#%%
trainHam = list(train.loc[train['type'] == 'ham', 'text'])
trainSpam = list(train.loc[train['type'] == 'spam', 'text'])
#%%

classifier1 = NaiveBayesForSpam()
classifier1.train(trainHam,trainSpam)

##  np.size(classifier1.likelihoods)
##  19078

#%%

classifier2 = NaiveBayesForSpam()
classifier2.train2(trainHam,trainSpam)

##  np.size(classifier2.likelihoods)
##  872

#%%
valMessages = validation['text']
valLabels = validation['type']

#%%
classifier1.score(valMessages, valLabels)

##
##(0.97599999999999998, array([[ 865.,   23.],
##        [   1.,  111.]]))

#1 ham classified as spam
#%%
classifier2.score(valMessages, valLabels)
##
##(0.98099999999999998, array([[ 864.,   17.],
##      [   2.,  117.]]))
# 2 ham classified as spam

#%%
#classifier2 is faster, because it uses less of the data, only spam keywords.
# it also performs because it doesn't include "unsure" words, e.g. "the" and "at"
# which would reduce the probability of spam if included (e.g. WIN the PRIZE at... ),
# only contains highly predictive words (e.g. win, prize)
#%%
#Q9 - 2 false positives, this could be reduced by changing the threshold.
# instead of 0.5 for spam, could increase to 0.7/0.8 etc.
#%%
testMessages = test['text']
testLabels = test['type']
#%%
#Note: being lazy, should probably use the train + val to retrain train2
classifier2.score(testMessages, testLabels)
#
#(0.97972972972972971, array([[ 1780.,    34.],
 #       [    8.,   250.]]))
#
