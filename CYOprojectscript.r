
## Harvard Edx - Diabetes Risk Prediction - CYO Project

## Poonam Quraishy - https://github.com/pqrshy/EdxCYO

##### Methods ####

# Install and load all the required libraries #

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(knitr)) install.packages("knitr")
if(!require(rpart)) install.packages("rpart")
if(!require(rpart.plot)) install.packages("rpart.plot")
if(!require(caret)) install.packages("caret")
if(!require(grid)) install.packages("grid")
if(!require(gridExtra)) install.packages("gridExtra")
if(!require(gbm)) install.packages("gbm")
if(!require(xgboost)) installed.packages("xgboost")
if(!require(e1071)) install.packages("e1071")
if(!require(corrplot)) install.packages("corrplot")
if(!require(MLeval)) install.packages("MLeval")
if(!require(ranger)) install.packages("ranger")
if(!require(quantmod)) install.packages("quantmod")
if(!require(kernlab)) install.packages("kernlab")
if(!require(markdown)) install.packages("markdown")
if(!require(rmarkdown)) install.packages("rmarkdown")
if(!require(stringr)) install.packages("stringr")
if(!require(pROC)) install.packages("pROC")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(gbm)) install.packages("gbm")
if(!require(dplyr)) install.packages("dplyr")
if(!require(MLeval)) install.packages("MLeval")
if(!require(caret)) install.packages("caret")
if(!require(xgboost)) install.packages("xgboost")
if(!require(pROC)) install.packages("pROC")
if(!require(class)) install.packages("class")
if(!require(ROCR)) install.packages("ROCR")
if(!require(randomForest)) install.packages("randomForest")
if(!require(reshape2)) install.packages("reshape2")
if(!require(mlbench)) install.packages("mLbench")


# Loading required libraries
library(tidyverse)
library(kableExtra)
library(tidyr)
library(knitr)
library(gbm)
library(caret)
library(MLeval)
library(rpart)
library(pROC)
library(rpart.plot)
library(grid)
library(gridExtra)
library(e1071)
library(xgboost)
library(corrplot)
library(mlbench)
library(MLeval)
library(quantmod)
library(kernlab)
library(ranger)
library(markdown)
library(rmarkdown)
library(stringr)
library(ggplot2)
library(pROC)
library(gbm)
library(dplyr)
library(caret)
library(xgboost)
library(class)
library(ROCR)
library(randomForest)
library(reshape2)

## The Pima Indians Diabetes dataset is available in the package mLbench.

data(PimaIndiansDiabetes)

################################################################################

### Exploratory Analysis

###############################################################################

pimadf <- PimaIndiansDiabetes

str(pimadf)

## First few rows

head(pimadf)

## Names of the columns 

colnames(pimadf)

## Summary of the dataset

summary(pimadf)

## Dimensions 

dim(pimadf)

## Check for NA's or missing values

sapply(pimadf, function(x) sum(is.na(x)))

## Exploring the response variable Diabetes

pimadf$diabetes <- factor(pimadf$diabetes)

# Diabetes - The sample has a high occurrence of positive diabetes diagnosis.

ggplot(pimadf,aes(diabetes,fill = diabetes)) +
  geom_bar() + 
  ggtitle("Distribution of diabetes variable")

## Plot of Correlations between all the predictor variables

corrmatx <- cor(pimadf[, -9])

corrplot.mixed(corrmatx,tl.pos = "lt")

## Numerical Representation

corrplot::corrplot(corrmatx, type = "lower", method = "number")

## Univariate Analysis 

bivar_plot <- function(bivar_name, bivar, data, output_var) {
  
  p_1 <- ggplot(data = data, aes(x = bivar, fill = output_var)) +
    geom_bar(stat='count', position='identity') +
    theme_bw() +
    labs( title = paste(bivar_name,"- Diabetes", sep =" "), x = bivar_name) +
    theme(plot.title = element_text(hjust = 0.5))
  
  plot(p_1)
}

for (x in 1:(ncol(pimadf)-1)) {
  bivar_plot(bivar_name = names(pimadf)[x], bivar = pimadf[,x], data = pimadf, output_var = pimadf[,'diabetes'])
}

## It is evident that high glucose levels lead to a higher chance of positive diabetes diagnosis.
## Mass/BMI increases also increase the chance of a positive diabetes diagnosis.
## Age over the age of 25 also increase the chance of diabetes diagnosis.
## There is no notable significant distinction among other variables to warrant further exploratory.

# Exploring the Age variable

p2 <- ggplot(pimadf, aes(age, fill = diabetes)) +
  geom_histogram(binwidth = 5) +
  theme(legend.position = "bottom") +
  ggtitle("Age of women Vs Diabetes")

p1 <- ggplot(pimadf, aes(x = diabetes, y = age,fill = diabetes)) +
  geom_boxplot() +
  theme(legend.position = "bottom") +
  ggtitle("Age of women Vs Diabetes")

gridExtra::grid.arrange(p1, p2, ncol = 2)

## Exploring the mass variable


m1 <- ggplot(pimadf, aes(mass, fill = diabetes)) +
  geom_histogram() +
  theme(legend.position = "bottom") +
  ggtitle("Mass of women Vs Diabetes")

m2 <- ggplot(pimadf, aes(x = diabetes, y = mass,fill = diabetes)) +
  geom_boxplot(binwidth = 30)  +
  theme(legend.position = "bottom") +
  ggtitle("Mass of women Vs Diabetes")

gridExtra::grid.arrange(m2, m1, ncol = 2)

## Exploring the Glucose variable

d1 <- ggplot(pimadf, aes(x = glucose, color = diabetes, fill = diabetes)) +
  geom_density(alpha = 0.8) +
  theme(legend.position = "bottom") +
  labs(x = "Glucose", y = "Density", title = "Density plot of glucose")

d2 <- ggplot(pimadf, aes(x = diabetes, y = glucose,fill = diabetes)) +
  geom_boxplot() +
  theme(legend.position = "bottom") +
  ggtitle("Glucose levels Vs Diabetes")

gridExtra::grid.arrange(d2, d1, ncol = 2)

###############################################################################

####  Modeling Approach ####

###############################################################################

## Split the Data into Training set consisting of 80% of the data and Testing set 

# consisting of 20% of the data.


partition <- caret::createDataPartition(y = pimadf$diabetes, times = 1, p = 0.8, list = FALSE)

# training set
train_set <- pimadf[partition,]

#test set
test_set <- pimadf[-partition,]

str(train_set)

summary(train_set)

################################################################################
######### Model 1 - Logistic Regression.  ##########
################################################################################

glm_model <- caret::train(diabetes ~., data = train_set,
                          method = "glm",
                          metric = "ROC",
                          tuneLength = 10,
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary),
                          preProcess = c("center","scale","pca"))


##  final ROC for the Logistic Regression 
glm_model$results[1,2]


################################################################################
####### Model 2 - Classification Tree.  ##########
################################################################################

tree_model <- rpart(diabetes~., data=train_set, method="class")
tree_model
rpart.plot(tree_model)

# Best Model Complexity Parameter - CP
plotcp(tree_model)

## The complexity parameter value of 0.015 was chosen since the relative error does 
# not decrease significantly was chosen as the Cp of the final tree.


# Model after Pruning the tree
tree_model <- rpart(diabetes~., data=train_set, method="class",cp=0.015)
rpart.plot(tree_model)

################################################################################
###### Model 3 - XGBOOST - Extreme Gradient Boosting  ######
################################################################################


xgb_grid  <-  expand.grid(
                         nrounds = 50,
                         eta = c(0.03),
                         max_depth = 1,
                         gamma = 0,
                         colsample_bytree = 0.6,
                         min_child_weight = 1,
                         subsample = 0.5
)

xgb_model <- caret::train(diabetes ~., data = train_set,
                          method = "xgbTree",
                          metric = "ROC",
                          tuneGrid=xgb_grid,
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary),
                          preProcess = c("center","scale","pca"))
xgb_model

xgb_model$results["ROC"]

################################################################################
###### Model 4 - K Nearest Neighbors (KNN)  ######
################################################################################


knn_model <- caret::train(diabetes ~., data = train_set,
                          method = "knn",
                          metric = "ROC",
                          tuneGrid = expand.grid(.k = c(3:10)),
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary),
                          preProcess = c("center","scale","pca"))

knn_model
plot(knn_model)

knn_model$results[7,2]


################################################################################
###### Model 5 - Support vector machine (SVM) ######
################################################################################


# Selecting the best parameters using the tune() function to do a grid search over 
# the supplied parameter ranges (C - cost, gamma), using the train set. 
# The range to gamma parameter is between 0.000001 and 0.1. 
# For cost parameter the range is from 0.1 until 10.


model_svmtune <- tune.svm(diabetes ~., data = train_set, gamma = 10^(-6:-1), cost = 10^(-1:1))
summary(model_svmtune) 

# As we can see the result show that the best parameters are Cost=10 and gamma=0.01.

svm_model  <- svm(diabetes ~., data = train_set, kernel = "radial", gamma = 0.01, cost = 10, probability = TRUE) 
summary(svm_model)



################################################################################
##### Prediction on the Test data set. #####
################################################################################

#### Model 1 - Logistic Regression


# prediction on Test data set
pred_glm <- predict(glm_model, test_set)

# Confusion Matrix 
cmx_glm <- confusionMatrix(pred_glm, test_set$diabetes, positive="pos")


pred_prob_glm <- predict(glm_model, test_set, type="prob")
# ROC value
roc_glm <- roc(test_set$diabetes, pred_prob_glm$pos)

# Confusion matrix 
cmx_glm

# ROC

roc_glm

# ROC Curve

rocur_glm <- caTools::colAUC(pred_prob_glm, test_set$diabetes, plotROC = T)

###### Model 2 - Classification tree


# prediction on the test data set.

pred_rpart <- predict(tree_model, test_set, type = "class")

# prediction probabilities
tree_prob_pred <- predict(tree_model, test_set, type = "prob")

# confusion matrix
cmx_tree <- confusionMatrix(test_set$diabetes, pred_rpart)

cmx_tree

rocur_tree <- caTools::colAUC(tree_prob_pred, test_set$diabetes, plotROC = T)

######## Model 3 - XGBOOST

# prediction on Test data set
pred_xgb <- predict(xgb_model, test_set)

# Confusion Matrix 
cmx_xgb <- confusionMatrix(pred_xgb, test_set$diabetes, positive="pos")


pred_prob_xgb <- predict(xgb_model, test_set, type="prob")
# ROC value
roc_xgb <- roc(test_set$diabetes, pred_prob_xgb$pos)

# Confusion matrix 
cmx_xgb

rocur_xgb <- caTools::colAUC(pred_prob_xgb, test_set$diabetes, plotROC = T)


######### Model 4 - KNN

# prediction on Test data set
pred_knn <- predict(knn_model, test_set)

# Confusion Matrix 
cmx_knn <- confusionMatrix(pred_knn, test_set$diabetes, positive="pos")

pred_prob_knn <- predict(knn_model, test_set, type="prob")
# ROC value
roc_knn <- roc(test_set$diabetes, pred_prob_knn$pos)

# Confusion matrix 
cmx_knn

rocur_knn <- caTools::colAUC(pred_prob_knn, test_set$diabetes, plotROC = T)

######### Model 5 - SVM

# prediction on the test data

par(mfrow = c(1,2))
pred_svm <- predict(svm_model, test_set)


svm_prob_pred <- predict(svm_model,test_set,probability = TRUE)


# Confusion matrix
cmx_svm <- confusionMatrix(test_set$diabetes, svm_prob_pred)

cmx_svm


###### Comparing the Test results of all the models.


test_glm <- c(cmx_glm$byClass['Sensitivity'], cmx_glm$byClass['Specificity'], cmx_glm$byClass['Precision'], 
                cmx_glm$byClass['Recall'], cmx_glm$byClass['F1'])

test_tree <- c(cmx_tree$byClass['Sensitivity'], cmx_tree$byClass['Specificity'], cmx_tree$byClass['Precision'], 
                  cmx_tree$byClass['Recall'], cmx_tree$byClass['F1'])


test_xgb <- c(cmx_xgb$byClass['Sensitivity'], cmx_xgb$byClass['Specificity'], cmx_xgb$byClass['Precision'], 
                cmx_xgb$byClass['Recall'], cmx_xgb$byClass['F1'])

test_knn <- c(cmx_knn$byClass['Sensitivity'], cmx_knn$byClass['Specificity'], cmx_knn$byClass['Precision'], 
                cmx_knn$byClass['Recall'], cmx_knn$byClass['F1'])

test_svm <- c(cmx_svm$byClass['Sensitivity'], cmx_svm$byClass['Specificity'], cmx_svm$byClass['Precision'], 
                cmx_svm$byClass['Recall'], cmx_svm$byClass['F1'])




test_results <- data.frame(rbind(test_glm, test_tree, test_knn, test_xgb, test_svm))
names(test_results) <- c("Sensitivity", "Specificity", "Precision", "Recall", "F1")
test_results


glm_accuracy <- confusionMatrix(pred_glm, test_set$diabetes)$overall['Accuracy']

tree_accuracy <- confusionMatrix(pred_rpart, test_set$diabetes)$overall['Accuracy']

xgb_accuracy <- confusionMatrix(pred_xgb, test_set$diabetes)$overall['Accuracy']

knn_accuracy <- confusionMatrix(pred_knn, test_set$diabetes)$overall['Accuracy']

svm_accuracy <- confusionMatrix(pred_svm, test_set$diabetes)$overall['Accuracy']

### Comparing the accuracy of all the models

accuracy <- data.frame(Model=c("Logistic Regression","Classification Tree","XGboost","K nearest neighbor (KNN)", "Support Vector Machine (SVM)"), 
                       Accuracy=c(glm_accuracy, tree_accuracy, xgb_accuracy, knn_accuracy, svm_accuracy ))


ggplot(accuracy,aes(x=Model,y=Accuracy)) + geom_bar(stat='identity') + theme_bw() + ggtitle('Comparison of Model Accuracy')+
  geom_col(fill = "coral1") +  geom_jitter(width=0.15) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# accuracy table
accuracy




































