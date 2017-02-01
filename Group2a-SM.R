library(class)

setwd("D:/Imperial MSc/Core Modules/Machine Learning/Assignments/Assignment 1")

set.seed(2017)

zScore <- function(data, meanVal, sdVal) {
    
    return((data - meanVal[col(data)]) / sdVal[col(data)])
}

whiteWines <- read.csv("winequality-white.csv", sep = ";")
# New binary column
whiteWines$GoodWine <- (whiteWines$quality >= 6)

whiteWines.X <- subset(whiteWines, select = -c(quality, GoodWine))
whiteWines.Y <- whiteWines$GoodWine

# Note: scale uses n - 1 as denomiator, i.e. sample standard deviation
#whiteWines.zScore <- as.data.frame(scale(subset(whiteWines, select = -c(quality, GoodWine))))
#whiteWines.zScore$GoodWine <- whiteWines$GoodWine

# Separate data into training (~40%), validation (~30%), and test (~30%)
trainSize <- round(0.4 * nrow(whiteWines))
inTrain <- sample.int(nrow(whiteWines), size = trainSize)

train.X <- whiteWines.X[inTrain, ]
train.Y <- whiteWines.Y[inTrain]
others.X <- whiteWines.X[-inTrain, ]
others.Y <- whiteWines.Y[-inTrain]

validationSize <- round(0.5 * nrow(others.X))
inValidation <- sample.int(nrow(others.X), size = validationSize)

validation.X <- others.X[inValidation, ]
validation.Y <- others.Y[inValidation]
test.X <- others.X[-inValidation, ]
test.Y <- others.Y[-inValidation]

trainValid.X <- rbind(train.X, validation.X)
trainValid.Y <- c(train.Y, validation.Y)

# Compute column mean and sd for train + validation set (for scaling purpose)
trainValid.X.mean <- colMeans(trainValid.X)
trainValid.X.sd <- apply(trainValid.X, 2, sd) # sd function uses n - 1 as denominator (sample sd)

train.X.zScore <- zScore(train.X, trainValid.X.mean, trainValid.X.sd)
validation.X.zScore <- zScore(validation.X, trainValid.X.mean, trainValid.X.sd)
test.X.zScore <- zScore(test.X, trainValid.X.mean, trainValid.X.sd)
trainValid.X.zScore <- zScore(trainValid.X, trainValid.X.mean, trainValid.X.sd)

validationPerf = data.frame(k = NULL, acc = NULL, sens = NULL, spec = NULL)

# Train 80 KNN classifiers and validate using validation data
for (i in 1:80) {
    knn.pred <- knn(train = train.X.zScore, 
                    test = validation.X.zScore, 
                    cl = train.Y, 
                    k = i)
    perfTable <- table(validation.Y, knn.pred)
    numTP <- perfTable["TRUE", "TRUE"]
    numTN <- perfTable["FALSE", "FALSE"]
    validationPerf <- rbind(validationPerf, 
                            data.frame(k = i, 
                                       acc = (numTP + numTN) / nrow(validation), 
                                       sens = numTP / (numTP + perfTable["TRUE", "FALSE"]), 
                                       spec = numTN / (numTN + perfTable["FALSE", "TRUE"])))
}

# Naive predictor accuracy (for benchmark)
sum(train.Y) / length(train.Y) # TRUE is the dominant class in training set
# Confusion Matrix of naive predictor
table(validation.Y, rep(TRUE, length(validation.Y)))["TRUE", "TRUE"] / length(validation.Y)

# Best K based on accuracy
bestK <- validationPerf[order(-validationPerf$acc), "k"][1]

# Test Phase: retrain using both train and validation data before testing
knn.pred.test <- knn(train = trainValid.X.zScore, 
                     test = test.X.zScore, 
                     cl = trainValid.Y, 
                     k = bestK)

# Confusion Matrix - Generalisation Error
(testPerf <- table(test.Y, knn.pred.test))
numTP <- testPerf["TRUE", "TRUE"]
numTN <- testPerf["FALSE", "FALSE"]
testPerf <- data.frame(k = bestK, 
                       acc = (numTP + numTN) / length(test.Y), 
                       sens = numTP / (numTP + testPerf["TRUE", "FALSE"]), 
                       spec = numTN / (numTN + testPerf["FALSE", "TRUE"]))

# Final step, train using all data for future use

# Scale using all data
whiteWines.X.zScore <- as.data.frame(scale(whiteWines.X))
whiteWines.knn <- knn(train = whiteWines.X.zScore, 
                      test = whiteWines.X.zScore, 
                      cl = whiteWines.Y, 
                      k = bestK)

