#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:47:11 2017

@author: cheungcecilia
"""

import pandas as pd
import numpy as np
import string
import random

data = pd.read_table("SMSSpamCollection.txt", delimiter = "	", header = None)

random.seed(123)

#create new df and copy values into new dataframe while removing punctuation and turn to lower caps

data_new = pd.DataFrame(index = range(0,5572), columns = range(0,2))

for i in range(0,len(data.iloc[:,1])):
    sentence = data.iloc[i,0]
    data_new.iloc[i,0] = sentence


for i in range(0, len(data.iloc[:,1])):
    line = data.iloc[i,1]
    new = line.translate(str.maketrans({a:None for a in string.punctuation}))
    new1 = new.lower()   
    data_new.iloc[i,1] = new1


#shuffle data

def shuffle(df):
    datashuffle = df.reindex(np.random.permutation(df.index))
    return datashuffle
    

data_sh = shuffle(data_new)

train_set = data_sh.iloc[0:2500, : ]
validation_set = data_sh.iloc[2500:5000, : ]
test_set = data_sh.iloc[5000:5572, : ]

#Naive Bayes classifier from scratch

class NaiveBayesForSpam:
    def train (self, hamMessages, spamMessages) :
        self.words = set (' '.join(hamMessages + spamMessages).split()) #create a set of words from both messages
        self.priors = np.zeros(2) #create empty array
        self.priors [0] = float(len(hamMessages)) / (len(hamMessages) + len(spamMessages)) #probability of getting ham messages
        self.priors [1] = 1.0 - self.priors[0] #probability of getting spam messaage
        self.likelihoods = []
        for i, w in enumerate(self.words):
            prob1 = (1.0 + len ([m for m in hamMessages if w in m])) / len( hamMessages )
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len ( spamMessages )
            self.likelihoods.append ([min (prob1 , 0.95) , min (prob2,0.95)])
        self.likelihoods = np.array (self.likelihoods).T
  
    def train2 ( self , hamMessages , spamMessages) :
        self.words = set (' '.join (hamMessages + spamMessages).split()) 
        self.priors = np.zeros(2)
        self.priors[0] = float(len(hamMessages)) / (len(hamMessages) + len(spamMessages) )
        self.priors [1] = 1.0 - self.priors[0]
        self.likelihoods = []
        spamkeywords = [ ]
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len([m for m in hamMessages if w in m])) /len( hamMessages )
            prob2 = (1.0 + len([m for m in spamMessages if w in m])) /len( spamMessages ) 
            if prob1*20 < prob2:
                self.likelihoods.append([min(prob1,0.95) ,min(prob2,0.95)])
                spamkeywords.append(w)
        self.words = spamkeywords
        self.likelihoods = np.array(self.likelihoods).T
    
    def predict (self, message) :
        posteriors = np.copy(self.priors)
        for i, w in enumerate (self.words):
            if w in message.lower(): # convert to lower−case
                posteriors *= self.likelihoods [: , i ] 
            else :
                posteriors *= np.ones(2) - self.likelihoods[:,i] 
            posteriors = posteriors / np.linalg.norm (posteriors) #normalise
        if posteriors [0] > 0.5:
            return ['ham', posteriors[0]]
        return ['spam', posteriors[1]] 
    
    def score ( self , messages , labels ) :
        confusion = np. zeros (4) . reshape (2 ,2) 
        for m, l in zip (messages, labels):
            if self.predict(m)[0] == 'ham' and l == 'ham': 
                confusion [0 ,0] += 1
            elif self.predict(m)[0] == 'ham' and l == 'spam': 
                confusion [0 ,1] += 1
            elif self.predict(m)[0] == 'spam' and l == 'ham': 
                confusion [1 ,0] += 1
            elif self.predict(m)[0] == 'spam' and l == 'spam': 
                confusion [1 ,1] += 1
        return (confusion[0,0] + confusion[1,1]) / float (confusion.sum() ) , confusion

#Explain the code: What is the purpose of each function? 
#What do ’train’ and ‘train2’ do, and what is the difference between them? 
#Where in the code is Bayes’ Theorem being applied?     

#train - train the naive bayes classifier using train data
#predict - predicts whether message is ham/spam
#score - calls predict function and outputs a confusion matrix and accuracy    
#difference between train and train2 -> train has a list of key spam words, so more selective when it comes to classification
     
                

#read all messages into a list

trainspam = train_set
trainspam = trainspam.drop(trainspam[trainspam[0] == 'ham'].index)

trainspamwords = []
for i in range(0, len(trainspam)):
    trainspamwords.append(trainspam.iloc[i, 1])

trainham = train_set
trainham = trainham.drop(trainham[trainham[0] == 'spam'].index)
    
trainhamwords = []
for i in range(0, len(trainham)):
    trainhamwords.append(trainham.iloc[i, 1])

    
#train using predefined functions
    
cldf1 = NaiveBayesForSpam()
cldf1.train(trainhamwords, trainspamwords)

cldf2 = NaiveBayesForSpam()
cldf2.train2(trainhamwords, trainspamwords)

#validation

valMessages = validation_set[1]
valType = validation_set[0]

cldf1.score(valMessages, valType)

#(0.98119999999999996, array([[ 2129.,    38.],
#                             [    9.,   324.]]))

cldf2.score(valMessages, valType)


#(0.97719999999999996, array([[ 2130.,    49.],
#                             [    8.,   313.]]))



#Why is the ‘train2’ classifier faster?
#Has the extra 'if' step which only runs if prob1*20 < prob2

#Why does it yield a better accuracy both on the training and the validation set?
#takes into account key spam words only


#How many false positives (ham messages classified as spam messages) did you get 
#in your validation set? How would you change the code to reduce false positives 
#at the expense of possibly having more false negatives (spam messages classified
#as ham messages)?
#38 for train, 49 for train2
#decrease threshold of the 'spam key word'
                                                         
                                                         

#run train2 classifier on test set

testMessages = test_set[1]
testType = test_set[0]

result = cldf2.score(testMessages, testType)

#(0.98426573426573427, array([[ 497.,    8.],
#                             [   1.,   66.]]))

test_confusionmatrix = pd.DataFrame(result[1])
test_confusionmatrix.columns = ['ham','spam']
test_confusionmatrix.index = ['ham','spam']
 
test_confusionmatrix


