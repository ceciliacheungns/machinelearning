# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:55:46 2017

@author: Nikhita Venkatesan
"""

import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split

seed = 12
np.random.seed(seed)

# Question 1 - Reading the data file
data = pd.read_table('SMSSpamCollection.txt', header = None, names = ['type', 'text'])

# Question 2 - Removing punctuation and changing characters to lower-case
translator = str.maketrans(string.ascii_uppercase, 
                           string.ascii_lowercase, 
                           string.punctuation + string.digits)

data['text'] = data['text'].str.translate(translator)

# Question 3 - Splitting data into a training, validation and test data set
train, others = train_test_split(data, train_size = 2500, random_state = seed)
validation, test = train_test_split(others, train_size = 1000, random_state = seed)

# Getting two lists of the ham and spam messages
hamMessages = list(train.loc[train['type'] == 'ham', 'text'])
spamMessages = list(train.loc[train['type'] == 'spam', 'text'])

messages = validation['text']
labels = validation['type']

# Question 4 - Naive Bayes Class-ifier
class NaiveBayesForSpam:
    def train(self, hamMessages, spamMessages):
        self.words = set(' '.join(hamMessages + spamMessages).split())
        self.priors = np.zeros(2)
        self.priors[0] = float(len(hamMessages)) / (len(hamMessages) + len(spamMessages))
        # probability of being a ham message
        self.priors[1] = 1.0 - self.priors[0]
        # probabiity of being a spam message
        self.likelihoods = []
        for i, w in enumerate(self.words): # for each word in the list of words
            prob1 = (1.0 + len([m for m in hamMessages if w in m])) / len(hamMessages)
            # ... divide number of ham messages with the word by the total number of messages
            prob2 = (1.0 + len([m for m in spamMessages if w in m])) / len(spamMessages)
            # ... divide number of spam messages with the word by the total number of messages
            self.likelihoods.append([min(prob1, 0.95), min(prob2, 0.95)])
            # probability ( ham or spam | certain word )
        self.likelihoods = np.array(self.likelihoods).T
        
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

    def predict(self, message):
        posteriors = np.copy(self.priors)
        for i, w in enumerate(self.words):
            if w in message.lower(): 
                posteriors *= self.likelihoods[:, i]
            else:
                posteriors *= np.ones(2) - self.likelihoods[:, i]
            posteriors = posteriors / np.linalg.norm(posteriors, ord = 1) 
        if posteriors[0] > 0.5:
            return ['ham', posteriors[0]]
        return ['spam', posteriors[1]]

    def score(self, messages, labels):
        confusion = np.zeros(4).reshape(2, 2)
        for m, l in zip(messages, labels):
            if self.predict(m)[0] == 'ham' and l == 'ham': # it is ham and we predict ham
                confusion[0, 0] += 1 # true negatives
            elif self.predict(m)[0] == 'ham' and l == 'spam': # it is spam but we predict ham
                confusion[0, 1] += 1 # false negatives
            elif self.predict(m)[0] == 'spam' and l == 'ham': # it is ham but we predict spam
                confusion[1, 0] += 1 # false positives
            elif self.predict(m)[0] == 'spam' and l == 'spam': # it is spam and we predict spam
                confusion[1, 1] += 1 # true positives
        return (confusion[0, 0] + confusion[1, 1]) / float(confusion.sum()), confusion
        # returns the accuracy rate and the confusion matrix

# Question 6 - Both classifiers trained   
NBtrain = NaiveBayesForSpam()
NBtrain.train(hamMessages,spamMessages)

NBtrain2 = NaiveBayesForSpam()
NBtrain2.train2(hamMessages,spamMessages)

# Question 7 - Performance of each classifier on the validation set
NBtrain.score(messages, labels) # 0.964999
# Confusion Matrix:
#[ 854, 13]
# [22, 111]

NBtrain2.score(messages, labels) # 0.973999
# Confusion Matrix:
#[ 873, 23]
# [3, 101]

# There is an improvement in accuracy which the train2 classifier is used on the validation set.
# The train classifier has less true positives and true negatives than the train2 classifier.
# The train2 classifier gives us more false positives than the train classifier.
# The train classifier gives us more false negatives than the train2 classifiers.