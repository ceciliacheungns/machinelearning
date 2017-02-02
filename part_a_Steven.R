library(data.table)
library(class)
library(caret)
library(ggplot2)
library(dplyr)

# ---------------------------------------------------------------------

## load data from source

wine.data <- fread("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")

# ---------------------------------------------------------------------

# Construct a column for "good" wine. The column takes values 1 => "good", 0 => "not good".
wine.data$Good.wine <- ifelse(wine.data$quality >= 6, 1, 0)

# Since we are using the Good.wine variable as outcome, we eliminate the "quality" 
# variable. Not eliminating it and using it as a feature will result in 
# erroneous predictions as it is highly correlated in the with "Goog.wine".

# The correlation can be seen in the following plot.
plot(wine.data$quality, wine.data$Good.wine)
cor(wine.data$quality, wine.data$Good.wine)
wine.data <- wine.data[, -"quality"]


# To perform a z-normalization on the data we calculate the mean and standard deviations
# in our data. The statistics will be calculated on the train and validation portion of
# our data. The same mean and standard deviation will then be used to z-normalize the
# test data.
# We therefore split the data before performing the calculations. We also set the seed in 
# order to make our results reproducable and shuffle the data to avoid sampling errors.

set.seed(146)
set.seed(155)
wine.shuffled <- wine.data[sample(nrow(wine.data)),]

# The desired split for the data is 40% for training, 30% for validation, and 30% for test set.

split <- c(train = 0.4, validate = 0.3, test = 0.3)
split.assignment <- sample(cut(seq(nrow(wine.shuffled)), nrow(wine.shuffled)*cumsum(c(0,split)), labels = names(split)))
result <- split(wine.shuffled, split.assignment)
addmargins(prop.table(table(split.assignment)))

train <- result$train
validate <- result$validate
test <- result$test

# We can now use the train and validate set to calculate the mean and standard deviation we 
# will use to normalize our data.
wine.z <- rbind(train, validate)
mu <- colMeans(wine.z[, -"Good.wine"])
st.dev <- apply(wine.z[, -"Good.wine"], 2, FUN = sd)

# Now we use the calculated means and standard deviations to z-normalize our data. We omit 
# the outcome variables from data as those should not be normalized. It should be 
# noted that when calculating the summary statistics of train.norm and validate.norm separately
# the mean and standard deviation will not be 0 and 1 respectively. This is because we are not
# using the entire data (cbind of train and validate) to calculate the summart statistics.

train.norm.intermediate <- sweep(train[, -"Good.wine"], 2, mu)
train.norm <- sweep(train.norm.intermediate, 2, st.dev, FUN = "/")

validate.norm.intermediate <- sweep(validate[, -"Good.wine"], 2, mu)
validate.norm <- sweep(validate.norm.intermediate, 2, st.dev, FUN = "/")

# We also normalize our test data as we will need it to test the performance of the best model.
# However, instead of using the mean and standard deviation of the test data, we normalize
# it using the previously calculated means and standard deviations.

test.norm.intermediate <- sweep(test[, -"Good.wine"], 2, mu)
test.norm <- sweep(test.norm.intermediate, 2, st.dev, FUN = "/")

# We now create a function to run the KNN algorithm a desired number of times. The function 
# takes the highest desired K as an argument and run the KNN algorithm with K from 1 to the
# highest desired K. Itreturns a list with the resulting predicted outcomes for each K.

knn.func <- function(kMax) {
  # create empty list
  results <- list()
  # run KNN algorithm and predict on the validation set, k times
  for (i in 1:kMax) {
    # create variable with assign, and store result 
    res <- assign(paste0("KNN", i), knn(train.norm, validate.norm, train.labels, k = i))
    results[[i]] <- res
  }
  results
}

# In order to run the algorithm we need to extract the outcomes variable from the training
# set and store it in a separate variable. The outcomes in the validation set also need to 
# be removed but will not be used in the KNN algorithm.

train.labels <- unlist(train[, "Good.wine"])

# We now run the previously created function with a maximum K of 80.

k <- 80
k80 <- knn.func(k)

# Now that we have run the model for multiple K's we create a function that evaluates the 
# models calculating the accuracy from a confusion matrix. Our function takes the output of
# knn.func as an argument and outputs 

knn.eval <- function(Knn) {
  # empty list to store results
  results <- list()
  for (i in 1:length(Knn)) {
    # create confusion matrix and extract "accuracy" for each model
    confMat <- confusionMatrix(Knn[[i]], validate.labels)
    results[[i]] <- c(i, confMat$overall[1])
  }
  results
}

# Before we run the function we need to extract the true outcomes from the validation set.

validate.labels <- unlist(validate[, "Good.wine"])

# We now run the function created above and store the results in a data frame. We then order
# the data by the misspecification error rate in ascending order.

eval <- knn.eval(k80)

df.eval <- data.frame(matrix(unlist(eval), nrow=length(eval), byrow=TRUE))
colnames(df.eval) <- c("K.Neighbors", "Accuracy")
eval.ordered <- df.eval[order(-df.eval$Accuracy),]
eval.ordered$Misspecification.Error <- 1 - eval.ordered$Accuracy

# In order to get the fitted values of the training data we run the previously created
# knn.func but modify it to use the training data as the second argument for the knn function.
# This will return the fitted values for our training data.

knn.func.training.fitted <- function(kMax) {
  # create empty list
  results <- list()
  # run KNN algorithm and predict on the validation set, k times
  for (i in 1:kMax) {
    # create variable with assign, and store result 
    res <- assign(paste0("KNN", i), knn(train.norm, train.norm, train.labels, k = i))
    results[[i]] <- res
  }
  results
}

# We run the function for the same K=80.

k <- 80
k80.train <- knn.func.training.fitted(k)

# We evaluate the output of the function using a modified version eval.func. function.

knn.eval.train <- function(Knn) {
  # empty list to store results
  results <- list()
  for (i in 1:length(Knn)) {
    # create confusion matrix and extract "accuracy" for each model
    confMat <- confusionMatrix(Knn[[i]], train.labels)
    results[[i]] <- c(i, confMat$overall[1])
  }
  results
}

# We now run the function created above and store the results in a data frame. We then order
# the data by the misspecification error rate in ascending order.

eval.train <- knn.eval.train(k80.train)

df.eval.train <- data.frame(matrix(unlist(eval.train), nrow=length(eval.train), byrow=TRUE))
colnames(df.eval.train) <- c("K.Neighbors", "Accuracy")
eval.fitted.ordered <- df.eval.train[order(-df.eval.train$Accuracy),]
eval.fitted.ordered$Misspecification.Error <- 1 - eval.fitted.ordered$Accuracy

# As we are interested in analyzing the performances of the different models on both the
# training and the validation data we create a function the plots both. The function outputs
# a graph with K on the x-axis and the misspecification rate on the y-axis. In addition,
# it plots the performance of the best model, which is the model with the lowest misspecification
# error on the validation set. To facilitate the plotting we merge the dataframes from our 
# evaluation data frames and change the column names to avoid confusion.

k.eval <- left_join(eval.fitted.ordered, eval.ordered, by = c("K.Neighbors" = "K.Neighbors"))
colnames(k.eval) <- c("K",
                      "Accuracy.Training",
                      "Misspecification.Training",
                      "Accuracy.Validation",
                      "Misspecification.Validation")

k.eval.ordered <- k.eval[order(-k.eval[, "Accuracy.Validation"]),]

test.labels <- unlist(test[, "Good.wine"])

plot.results <- function(ordered.Ks) {
  # extract best K
  K.best <- ordered.Ks[1,1]
  # run KNN with best K
  bestK.fit <- knn(train.norm, test.norm, train.labels, k = K.best)
  eval.bestK <- confusionMatrix(bestK.fit, test.labels)
  acc <- eval.bestK$overall[1]
  print(acc)
  
  g <- ggplot(data = ordered.Ks, aes(x = K)) + 
    geom_line(aes(y = Misspecification.Validation)) +
    geom_line(aes(y = Misspecification.Training)) +
    geom_point(aes(x = K.best, y = (1 - acc))) +
    labs(title = "Performance Comparison", x = "Number of K", y = "Misspecification Error") +
    scale_y_continuous(breaks = c(0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)) +
    theme(legend.title=element_blank()) +
    scale_color_manual(labels = c("Validation", "Training"), values = c("blue", "red"))
  
  g
}

plot.results(k.eval.ordered)

# We store the results from our test.
save(k.eval.ordered ,file="performance_evaluation.Rda")