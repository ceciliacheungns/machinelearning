# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:15:03 2017

@author: Steven
"""
import pandas as pd
import numpy as np
import io   # for opening the text file encoded in utf-8
import re   # regular expression
import string   # to get character list
import random   # to set seed


# load data using utf8 encoding
data = io.open('SMSSpamCollection','r',encoding='utf8')
lines = data.readlines()

# put each line into a list
linesStripped = []
for i in range(len(lines)):
    linesStripped.append(lines[i].strip('\n'))

# separate each line and create a list of list with results
linesStr = []
for i in range(len(linesStripped)):
    linesStr.append(re.split(r'\t+', linesStripped[i]))

# store data in data frame
dframe = pd.DataFrame(linesStr, columns = ['Category', 'Message'])

# remove punctuation and digits be creating aregex function applied 
# to whole data frame
removeChar = string.punctuation + string.digits
rgx = re.compile('[%s]' % removeChar)

def removeCharFunc(x):
    return rgx.sub('', x)

dframe['Message'] = dframe['Message'].apply(removeCharFunc, 1)

# lower case all characters
dframe['Message'] = dframe['Message'].str.lower()

# Shuffle data and split into train, validate, test
random.seed(42)

dframeShuffle = dframe.reindex(np.random.permutation(dframe.index))
trainSplit = range(2500)
validationSplit = range(len(trainSplit), len(trainSplit) + 1000)
testSplit = range(validationSplit[-1] + 1, len(dframeShuffle))

train = dframeShuffle.iloc[trainSplit]
validate = dframeShuffle.iloc[validationSplit]
test = dframeShuffle.iloc[testSplit]

'''-------------------------------------------------------------------------'''

'''Code for the classifier'''

class NaiveBayesForSpam:
    def train (self, hamMessages, spamMessages):
        # train method takes list of ham and spam messages!!!
        self.words = set(' '.join (hamMessages + spamMessages).split())
        self.priors = np.zeros(2)
        self.priors [0] = float(len(hamMessages)) / (len(hamMessages) +len(spamMessages))
        self.priors [1] = 1.0 - self.priors [0]
        self.likelihoods = []
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len ([m for m in hamMessages if w in m])) / len(hamMessages)
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len(spamMessages)
            self.likelihoods.append ([min (prob1 , 0.95) , min (prob2 ,0.95) ]) # 0.95 in case a word is in all the messages
        self.likelihoods = np.array (self.likelihoods).T

    def train2 (self, hamMessages, spamMessages):
        self.words = set(' '.join (hamMessages + spamMessages).split())
        self.priors = np.zeros(2)
        self.priors [0] = float (len (hamMessages)) / (len (hamMessages) + len(spamMessages))
        self.priors [1] = 1.0 - self.priors [0]
        self.likelihoods = []
        spamkeywords = []
        
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len ([m for m in hamMessages if w in m])) / len(hamMessages)
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len(spamMessages)
            
            if prob1 * 20 < prob2:  # avoids adding words that come up frequently in a mail (those hold little meaning)
                                    # in contrast, words like "viagra" almost certainly appear less frequently in ham and will then we added to the list of keywords
                self.likelihoods.append([min(prob1 , 0.95) , min (prob2 , 0.95) ])
                spamkeywords.append (w)
        self.words = spamkeywords
        self.likelihoods = np.array (self.likelihoods).T
    
    def predict (self, message):
        posteriors = np.copy (self.priors)
        for i, w in enumerate (self.words):
            if w in message.lower(): # convert to lower case
                posteriors *= self.likelihoods [:, i]
            else:
                posteriors *= np.ones(2) - self.likelihoods[:,i]
            posteriors =  posteriors / np.linalg.norm(posteriors, ord = 1) # normalise
        if posteriors[0] > 0.5:   # threshold!!!!
            return ['ham', posteriors [0]]
        return ['spam', posteriors [1]]
        
    def score (self, messages ,labels ):
        confusion = np. zeros (4) . reshape (2 ,2)
        for m, l in zip(messages, labels):
            if self.predict(m)[0] == 'ham' and l == 'ham':
                confusion [0 ,0] += 1
            elif self.predict(m)[0] == 'ham' and l == 'spam':
                confusion [0 ,1] += 1
            elif self.predict(m)[0] == 'spam' and l == 'ham':
                confusion [1 ,0] += 1
            elif self.predict(m)[0] == 'spam' and l == 'spam':
                confusion [1 ,1] += 1
        return (confusion[0,0] + confusion[1,1]) / float (confusion.sum()), confusion

'''-------------------------------------------------------------------------'''

'''Run models'''

'''7.'''   
# Initiate models
nbTrain = NaiveBayesForSpam()
nbTrain2 = NaiveBayesForSpam()

# separate ham and spam models
hamTrain = train[train['Category'] == 'ham']['Message']
spamTrain = train[train['Category'] == 'spam']['Message']

# train model with "train"
nbTrain.train(list(hamTrain), list(spamTrain))
# make predictions on validation set
nbTrain.score(list(validate['Message']), list(validate['Category']))


# train model with "train2"
nbTrain2.train2(list(hamTrain), list(spamTrain))
# make predictions on validation set
nbTrain2.score(list(validate['Message']), list(validate['Category']))

'''8.'''
nbTrain.score(list(train['Message']), list(train['Category']))
nbTrain2.score(list(train['Message']), list(train['Category']))

# The train2 classifier is faster because it uses only keywords, i.e. words that carry
# meaning and are more indicative of the nature of a message. This means the "score" 
# method needs to compare the content of each message to less words, making the algorithm
# faster.

# It yields better accuracy, although only slightly, because it only uses relevant words
# to classify messages. This means less noise in our classifier.

# 

'''9.'''
nbTrain2.score(list(validate['Message']), list(validate['Category']))
# output 
# 854   31
# 1     114
# We got 31 false positives. 
# We could reduce the theshold in the predict method. This will make it harder
# for our classifier to classify a message as spam.


'''10.'''
# we train the "train2" classifier on the entire available data (train + validation)
trainCompleteSplit = range(0,3500)
trainComplete = dframeShuffle.iloc[trainCompleteSplit]

hamComplete = trainComplete[trainComplete['Category'] == 'ham']['Message']
spamComplete = trainComplete[trainComplete['Category'] == 'spam']['Message']

nbTrainFinal = NaiveBayesForSpam()
nbTrainFinal.train2(list(hamComplete), list(spamComplete))

# make predictions on test set
nbTrainFinal.score(list(test['Message']), list(test['Category']))
# output
# 1778  54
# 3     239


'''Random chunks to test code'''
a = ['hello world', 'i love you', 'you buy', 'you you']
# set(' '.join (['Hello World', 'i love you'] + ['You suck', 'buy viagra']).split())
              
              
