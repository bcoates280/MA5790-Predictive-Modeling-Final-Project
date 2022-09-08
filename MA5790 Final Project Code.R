#####################################################
#MA 5790 - Final Project: Startup Classification
#####################################################

library(tidyverse)
library(dplyr)
library(mlbench)
library(caret)
library(AppliedPredictiveModeling)
library(e1071)
library(corrplot)
library(pROC)
library(MASS)
library(mda)
library(nnet)
library(earth)
library(kernlab)
library(klaR)
library(svmpath)

#load data
startups <- read.csv("./StartupData.csv")
attach(startups)

#Pre-Processing
startups <- filter(startups, country_code == "USA", founded_at > as.Date("1994-12-31"), state_code != "",
                   funding_rounds > 0, category_code != "", amt_raised > 0, time_to_first_milestone != "NA",
                   time_to_last_milestone != "NA", time_to_first_round >= 0, time_to_last_round >= 0)
startups$status <- as.factor(startups$status)
startups <- filter(startups, status != "operating")
levels(startups$status) <- c("Success","Failure","Success","Success")
startups$has_VC <- as.numeric(startups$has_VC)
startups$has_angel <- as.numeric(startups$has_angel)
startups$has_roundA <- as.numeric(startups$has_roundA)
startups$has_roundB <- as.numeric(startups$has_roundB)
startups$has_roundC <- as.numeric(startups$has_roundC)
startups$has_debtPE <- as.numeric(startups$has_debtPE)
startups$time_to_first_milestone <- as.numeric(startups$time_to_first_milestone)
startups$time_to_last_milestone <- as.numeric(startups$time_to_last_milestone)

#state dummy variables
startups$state_code <- as.factor(as.character(ifelse(startups$state_code == 'CA', yes = "CA", no = ifelse(startups$state_code == 'NY', 
                      yes = "NY", no = ifelse(startups$state_code == 'MA', 
                      yes = "MA", no = ifelse(startups$state_code == 'TX', yes = "TX", no = "other"))))))

#industry category dummy variables
startups$category_code <- as.factor(as.character(ifelse(startups$category_code == 'software', yes = "software", 
                            no = ifelse(startups$category_code == 'web', yes = "web", 
                            no = ifelse(startups$category_code == 'mobile', yes = "mobile", 
                            no = ifelse(startups$category_code == 'enterprise', yes = "enterprise", 
                            no = ifelse(startups$category_code == 'advertising', yes = "advertising",
                            no = ifelse(startups$category_code == 'games_video', yes = "games_video",
                            no = ifelse(startups$category_code == 'ecommerce', yes = "ecommerce",
                            no = ifelse(startups$category_code == 'consulting', yes = "consulting", 
                            no = ifelse(startups$category_code == 'biotech', yes = "biotech",    
                            no = "other")))))))))))

dmy <- dummyVars(formula = ~ state_code + category_code, 
                 data = startups, 
                 fullRank = TRUE)
dummy.vars <- data.frame(predict(dmy, newdata = startups))
head(dummy.vars, n = 10)

#Combined Dataset
startups.predictors <- cbind(dummy.vars, startups[12:24])
summary(startups.predictors)
startups.response <- startups[3]
summary(startups)


##################################
# Data Exploration
##################################

#Split data into categorical and quantitative group
startupsColNum <- cbind(startups.predictors[,14:17], startups.predictors[,24:26])
head(startupsColNum)
summary(startupsColNum)

startupsCatVar <- cbind(startups.predictors[,1:13], startups.predictors[,18:23])
head(startupsCatVar)

##################################
# Numerical Predictor Examination
##################################
#Create histograms of numeric variables
par(mfrow = c(3,3))
hist(startupsColNum$time_to_first_round, main = "Time to First Round", xlab = "Years")
hist(startupsColNum$time_to_last_round, main = "Time to Last Round", xlab = "Years")
hist(startupsColNum$amt_raised, main = "Amount Raised", xlab = "$USD")
hist(startupsColNum$funding_rounds, main = "Number of Funding Rounds", xlab = "Number")
hist(startupsColNum$milestones, main = "Number of Milestones", xlab = "Number")
hist(startupsColNum$time_to_first_milestone, main = "Time to First Milestone", xlab = "Years")
hist(startupsColNum$time_to_last_milestone, main = "Time to Last Milestone", xlab = "Years")

#Create Scatter plot matrix to examine relationships between predictors
panel.cor <- function(x , y, digits = 2, prefix = "", cex.cor, ...) {
  usr <- par("usr")
  on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  Cor <- cor(x, y) # Remove abs function if desired
  txt <- paste0(prefix, format(c(Cor, 0.123456789), digits = digits)[1])
  if(missing(cex.cor)) {
    cex.cor <- 0.4 / strwidth(txt)
  }
  text(0.5, 0.5, txt,
       cex = 1 + cex.cor * Cor) # Resize the text by level of correlation
}

pairs(startupsColNum,
      data = startupsColNum,
      upper.panel = panel.cor,
      lower.panel = panel.smooth)

#Calculate correlation between predictors
correlation <- cor(startupsColNum)
correlation
corrplot(correlation)

#Find high correlation predictors
highcor <- findCorrelation(correlation, 0.7)
colnames(startupsColNum)[highcor]

#Remove highly correlated predictors
startupsColNum.New <- startupsColNum[, -highcor]
dim(startupsColNum.New)
corrnew <- cor(startupsColNum.New)
corrplot(corrnew)

#######################################
# Categorical Exploration
#######################################
#recreate factors for funding round dummy variables
startupsCatVar$has_VC <- as.factor(startups$has_VC)
startupsCatVar$has_angel <- as.factor(startups$has_angel)
startupsCatVar$has_roundA <- as.factor(startups$has_roundA)
startupsCatVar$has_roundB <- as.factor(startups$has_roundB)
startupsCatVar$has_roundC <- as.factor(startups$has_roundC)
startupsCatVar$has_debtPE <- as.factor(startups$has_debtPE)

#Create bar plots of categorical data
par(mfrow = c(3,3))
plot(startupsCatVar$has_VC, main = "Has VC")
plot(startupsCatVar$has_angel, main = "Has Angel Investor")
plot(startupsCatVar$has_roundA, main = "Has Round A")
plot(startupsCatVar$has_roundB, main = "Has Round B")
plot(startupsCatVar$has_roundC, main = "Has Round C")
plot(startupsCatVar$has_debtPE, main = "Has Debt PE")
plot(startups$state_code, main = "State")
plot(startups$category_code, main = "Industry Category")
plot(startups.response, main = "Startup Status")

#Reset to numeric
startupsCatVar$has_VC <- as.numeric(startups$has_VC)
startupsCatVar$has_angel <- as.numeric(startups$has_angel)
startupsCatVar$has_roundA <- as.numeric(startups$has_roundA)
startupsCatVar$has_roundB <- as.numeric(startups$has_roundB)
startupsCatVar$has_roundC <- as.numeric(startups$has_roundC)
startupsCatVar$has_debtPE <- as.numeric(startups$has_debtPE)

#Find Near Zero or Zero Variance Predictors
nzv <- nearZeroVar(startupsCatVar)
nzv

#Predictors is_TX, is_ecommerce, and is_consulting were found to be nzv
#Remove these predictors from data set
startupsCatVar.New <- startupsCatVar[, -nzv]
head(startupsCatVar.New)

#state dummy variables
startups$state_code <- as.factor(as.character(ifelse(startups$state_code == 'CA', yes = "CA", no = ifelse(startups$state_code == 'NY', 
                                    yes = "NY", no = ifelse(startups$state_code == 'MA', 
                                    yes = "MA", no = ifelse(startups$state_code == 'TX', yes = "other", no = "other"))))))

#industry category dummy variables
startups$category_code <- as.factor(as.character(ifelse(startups$category_code == 'software', yes = "software", 
                                    no = ifelse(startups$category_code == 'web', yes = "web", 
                                    no = ifelse(startups$category_code == 'mobile', yes = "mobile", 
                                    no = ifelse(startups$category_code == 'enterprise', yes = "enterprise", 
                                    no = ifelse(startups$category_code == 'advertising', yes = "advertising",
                                    no = ifelse(startups$category_code == 'games_video', yes = "games_video",
                                    no = ifelse(startups$category_code == 'ecommerce', yes = "other",
                                    no = ifelse(startups$category_code == 'consulting', yes = "other", 
                                    no = ifelse(startups$category_code == 'biotech', yes = "biotech",    
                                    no = "other")))))))))))

dmy <- dummyVars(formula = ~ state_code + category_code, 
                 data = startups, 
                 fullRank = TRUE)
dummy.vars <- data.frame(predict(dmy, newdata = startups))
head(dummy.vars, n = 10)

startupsCatVar.New <- cbind(startupsCatVar.New[,11:16], dummy.vars[,1:10])

###############################
# Data Transformations
###############################
#First calculate skewness values for untransformed numerical predictors
skewnessval <- apply(startupsColNum.New, 2, skewness)
skewnessval

#Heavy skewness present in each variable except time to first milestone
#Need to center and scale variables first, cannot use Box Cox method due to =< 0 values, must add constant
#Add constant to all numerical predictors equal to the absolute value of the minimum value + 1
startupsColNum.add <- startupsColNum.New + 21
summary(startupsColNum.add)

startupsColNum.Trans <- preProcess(startupsColNum.add, method = c("center", "scale","BoxCox"))
startupsColNum.Trans

#Apply Transformations and review new skewness
Transformed <- predict(startupsColNum.Trans, startupsColNum.add)
skewtrans <- apply(Transformed, 2, skewness)
skewtrans

#Apply spatial sign transformation to lessen outlier effect
spatial.startups <- as.data.frame(spatialSign(Transformed))
skewspatial <- apply(spatial.startups, 2, skewness)
skewspatial

boxplot(spatial.startups)

#Join numerical and categorical variables to new predictor set
startups.predictors.clean <- cbind.data.frame(spatial.startups, startupsCatVar.New)
head(startups.predictors.clean)

###############################
# Random Sampling
###############################

#Use stratified random splits based on classes of response variable
#Set random number seed
set.seed(20)

#Create training set using 60% of data
trainingRows <- createDataPartition(as.matrix(startups.response), p = 0.6, list = FALSE)
head(trainingRows)
nrow(trainingRows)

trainPredictors <- startups.predictors.clean[trainingRows, ]
trainClasses <- startups.response[trainingRows]
length(trainClasses)
dim(trainPredictors)

traindata <- cbind(trainPredictors, trainClasses)
summary(traindata)

#Create test set of data
testPredictors <- startups.predictors.clean[-trainingRows, ]
testClasses <- startups.response[-trainingRows]

testdata <- cbind(testPredictors, testClasses)

#resampling
ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

#######################
#Logistic Regression
#######################
set.seed(200)
lrTrain <- train(traindata[,1:21],
                y = traindata$trainClasses,
                method = "glm",
                metric = "ROC",
                trControl = ctrl)
lrTrain
head(lrTrain$pred)
length(lrTrain$pred[,1])

#confusion matrix - training data
confusionMatrix(data = lrTrain$pred$pred,
                reference = lrTrain$pred$obs,
                positive = "Success")

#ROC curve - training data
trainROC.lr <- roc(response = lrTrain$pred$obs,
               predictor = lrTrain$pred$Success,
               levels = rev(levels(lrTrain$pred$obs)))
par(mfrow = c(1,1))
plot(trainROC.lr, legacy.axes = TRUE, main = "Training Data ROC Curve: Logistic Regression")
auc(trainROC.lr)

#testing data
predict.lr <- predict(lrTrain, testdata[,1:21], type="prob")
testPredictors$lrprob <- predict.lr[,"Success"]
testPredictors$lr.class <- predict(lrTrain, testdata[,1:21])
testPredictors$obs <- as.factor(testClasses)

#confusion matrix - testing data
confusionMatrix(data = testPredictors$lr.class,
                reference = testPredictors$obs,
                positive = "Success")

#ROC Curve- testing data
testROC.lr <- roc(response = testPredictors$obs,
                   predictor = testPredictors$lrprob,
                   levels = rev(levels(testPredictors$obs)))
plot(testROC.lr, legacy.axes = TRUE, main = "Testing Data ROC Curve: Logistic Regression")
auc(testROC.lr)

#Most important variables
glmnImp <- varImp(lrTrain, scale = FALSE)
glmnImp
plot(glmnImp, top = 9, scales = list(y = list(cex = .95)), main = "Most Important Variables")


#################################
#Linear Discriminant Analysis
#################################
set.seed(200)
LDAtrain <- train(traindata[,1:21],
                  y = traindata$trainClasses,
                 method = "lda",
                 metric = "ROC",
                 trControl = ctrl)
LDAtrain
head(LDAtrain$pred)
length(LDAtrain$pred[,1])

#confusion matrix - training data
confusionMatrix(data = LDAtrain$pred$pred,
                reference = LDAtrain$pred$obs,
                positive = "Success")

#ROC curve - training data
trainROC.LDA <- roc(response = LDAtrain$pred$obs,
                   predictor = LDAtrain$pred$Success,
                   levels = rev(levels(LDAtrain$pred$obs)))
plot(trainROC.LDA, legacy.axes = TRUE, main = "Training Data ROC Curve: Linear Discriminant Analysis")
auc(trainROC.LDA)

#testing data
predict.LDA <- predict(LDAtrain, testdata[,1:21], type="prob")
testPredictors$LDAprob <- predict.LDA[,"Success"]
testPredictors$LDA.class <- predict(LDAtrain, testdata[,1:21])
testPredictors$obs <- as.factor(testClasses)

#confusion matrix - testing data
confusionMatrix(data = testPredictors$LDA.class,
                reference = testPredictors$obs,
                positive = "Success")

#ROC Curve - testing data
testROC.LDA <- roc(response = testPredictors$obs,
                  predictor = testPredictors$LDAprob,
                  levels = rev(levels(testPredictors$obs)))
plot(testROC.LDA, legacy.axes = TRUE, main = "Testing Data ROC Curve: Linear Discriminant Analysis")
auc(testROC.LDA)

########################################################
#Partial Least Squares Linear Discriminant Analysis
########################################################
set.seed(200)
PLSDAtrain <- train(traindata[,1:21],
                    y = traindata$trainClasses,
                 method = "pls",
                 tuneGrid = expand.grid(.ncomp = 1:10),
                 metric = "ROC",
                 trControl = ctrl)
PLSDAtrain
plot(PLSDAtrain, main = "PLSDA Tuning Parameters")
head(PLSDAtrain$pred)
length(PLSDAtrain$pred[,1])

#confusion matrix - training data
confusionMatrix(data = PLSDAtrain$pred$pred,
                reference = PLSDAtrain$pred$obs,
                positive = "Success")

#ROC curve - training data
trainROC.PLSDA <- roc(response = PLSDAtrain$pred$obs,
                    predictor = PLSDAtrain$pred$Success,
                    levels = rev(levels(PLSDAtrain$pred$obs)))
plot(trainROC.PLSDA, legacy.axes = TRUE, main = "Training Data ROC Curve: PLSDA")
auc(trainROC.PLSDA)

#testing data
predict.PLSDA <- predict(PLSDAtrain, testdata[,1:21], type="prob")
testPredictors$PLSDAprob <- predict.PLSDA[,"Success"]
testPredictors$PLSDA.class <- predict(PLSDAtrain, testdata[,1:21])
testPredictors$obs <- as.factor(testClasses)

#confusion matrix - testing data
confusionMatrix(data = testPredictors$PLSDA.class,
                reference = testPredictors$obs,
                positive = "Success")

#ROC Curve - testing data
testROC.PLSDA <- roc(response = testPredictors$obs,
                   predictor = testPredictors$PLSDAprob,
                   levels = rev(levels(testPredictors$obs)))
plot(testROC.PLSDA, legacy.axes = TRUE, main = "Testing Data ROC Curve: PLSDA")
auc(testROC.PLSDA)

#######################
#Penalized Model
#######################
set.seed(476)
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))

glmnTrain <- train(x=traindata[,1:21],
                   y = traindata$trainClasses,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   metric = "ROC",
                   trControl = ctrl)
glmnTrain
plot(glmnTrain, main = "GLMN Tuning Parameters")
head(glmnTrain$pred)
length(glmnTrain$pred[,1])

#confusion matrix - training data
confusionMatrix(data = glmnTrain$pred$pred,
                reference = glmnTrain$pred$obs,
                positive = "Success")

#ROC curve - training data
trainROC.glmn <- roc(response = glmnTrain$pred$obs,
                     predictor = glmnTrain$pred$Success,
                     levels = rev(levels(glmnTrain$pred$obs)))
plot(trainROC.glmn, legacy.axes = TRUE, main = "Training Data ROC Curve: Penalized Model")
auc(trainROC.glmn)

#testing data
predict.glmn <- predict(glmnTrain, testdata[,1:21], type="prob")
testPredictors$glmnprob <- predict.glmn[,"Success"]
testPredictors$glmn.class <- predict(glmnTrain, testdata[,1:21])
testPredictors$obs <- as.factor(testClasses)

#confusion matrix - testing data
confusionMatrix(data = testPredictors$glmn.class,
                reference = testPredictors$obs,
                positive = "Success")

#ROC Curve - testing data
testROC.glmn <- roc(response = testPredictors$obs,
                    predictor = testPredictors$glmnprob,
                    levels = rev(levels(testPredictors$obs)))
plot(testROC.glmn, legacy.axes = TRUE, main = "Testing Data ROC Curve: Penalized Models")
auc(testROC.glmn)

#####################################
# Nonlinear Discriminant Analysis
#####################################
set.seed(200)
MDAtrain <- train(traindata[,1:21],
                y = traindata$trainClasses,
                method = "mda",
                metric = "ROC",
                tuneGrid = expand.grid(.subclasses = 1:3),
                trControl = ctrl)
MDAtrain
plot(MDAtrain, main = "MDA Tuning Parameters")
head(MDAtrain$pred)
length(MDAtrain$pred[,1])

#confusion matrix - training data
confusionMatrix(data = MDAtrain$pred$pred,
                reference = MDAtrain$pred$obs,
                positive = "Success")

#ROC curve - training data
trainROC.MDA <- roc(response = MDAtrain$pred$obs,
                      predictor = MDAtrain$pred$Success,
                      levels = rev(levels(MDAtrain$pred$obs)))
plot(trainROC.MDA, legacy.axes = TRUE, main = "Training Data ROC Curve: MDA")
auc(trainROC.MDA)

#testing data
predict.MDA <- predict(MDAtrain, testdata[,1:21], type="prob")
testPredictors$MDAprob <- predict.MDA[,"Success"]
testPredictors$MDA.class <- predict(MDAtrain, testdata[,1:21])
testPredictors$obs <- as.factor(testClasses)

#confusion matrix - testing data
confusionMatrix(data = testPredictors$MDA.class,
                reference = testPredictors$obs,
                positive = "Success")

#ROC Curve - testing data
testROC.MDA <- roc(response = testPredictors$obs,
                     predictor = testPredictors$MDAprob,
                     levels = rev(levels(testPredictors$obs)))
plot(testROC.MDA, legacy.axes = TRUE, main = "Testing Data ROC Curve: MDA")
auc(testROC.MDA)

##########################
# Neural Network Model
##########################
set.seed(200)
#Set training parameters
nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, 0.1, 0.3, 0.5, 1))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (21 + 1) + (maxSize + 1)*2)

NNettrain <- train(traindata[,1:21],
                   y = traindata$trainClasses,
                   method = "nnet",
                   metric = "ROC",
                   tuneGrid = nnetGrid,
                   trace = FALSE,
                   maxit = 2000,
                   maxNWts = numWts,
                   trControl = ctrl)
NNettrain
plot(NNettrain, main = "Neural Network")
head(NNettrain$pred)
length(NNettrain$pred[,1])

#confusion matrix - training data
confusionMatrix(data = NNettrain$pred$pred,
                reference = NNettrain$pred$obs,
                positive = "Success")

#ROC curve - training data
trainROC.NNet <- roc(response = NNettrain$pred$obs,
                    predictor = NNettrain$pred$Success,
                    levels = rev(levels(NNettrain$pred$obs)))
plot(trainROC.NNet, legacy.axes = TRUE, main = "Training Data ROC Curve: Neural Net")
auc(trainROC.NNet)

#testing data
predict.NNet <- predict(NNettrain, testdata[,1:21], type="prob")
testPredictors$NNetprob <- predict.NNet[,"Success"]
testPredictors$NNet.class <- predict(NNettrain, testdata[,1:21])
testPredictors$obs <- as.factor(testClasses)

#confusion matrix - testing data
confusionMatrix(data = testPredictors$NNet.class,
                reference = testPredictors$obs,
                positive = "Success")

#ROC Curve - testing data
testROC.NNet <- roc(response = testPredictors$obs,
                   predictor = testPredictors$NNetprob,
                   levels = rev(levels(testPredictors$obs)))
plot(testROC.NNet, legacy.axes = TRUE, main = "Testing Data ROC Curve: NNet")
auc(testROC.NNet)

################################
# Flexible Discriminant Analysis
################################
set.seed(200)
#Define training grid
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)

FDAtrain <- train(traindata[,1:21],
                  y = traindata$trainClasses,
                  method = "fda",
                  metric = "ROC",
                  tuneGrid = marsGrid,
                  trControl = ctrl)
FDAtrain
plot(FDAtrain, main = "Flexible Discriminant Analysis")
head(FDAtrain$pred)
length(FDAtrain$pred[,1])

#confusion matrix - training data
confusionMatrix(data = FDAtrain$pred$pred,
                reference = FDAtrain$pred$obs,
                positive = "Success")

#ROC curve - training data
trainROC.FDA <- roc(response = FDAtrain$pred$obs,
                     predictor = FDAtrain$pred$Success,
                     levels = rev(levels(FDAtrain$pred$obs)))
plot(trainROC.FDA, legacy.axes = TRUE, main = "Training Data ROC Curve: FDA")
auc(trainROC.FDA)

#testing data
predict.FDA <- predict(FDAtrain, testdata[,1:21], type="prob")
testPredictors$FDAprob <- predict.FDA[,"Success"]
testPredictors$FDA.class <- predict(FDAtrain, testdata[,1:21])
testPredictors$obs <- as.factor(testClasses)

#confusion matrix - testing data
confusionMatrix(data = testPredictors$FDA.class,
                reference = testPredictors$obs,
                positive = "Success")

#ROC Curve - testing data
testROC.FDA <- roc(response = testPredictors$obs,
                    predictor = testPredictors$FDAprob,
                    levels = rev(levels(testPredictors$obs)))
plot(testROC.FDA, legacy.axes = TRUE, main = "Testing Data ROC Curve: FDA")
auc(testROC.FDA)

###########################
# Support Vector Machine
###########################
set.seed(200)
#Define svmgrid
sigmaRangeReduced <- sigest(as.matrix(traindata[,1:21]))
svmGrid <- expand.grid(.sigma = sigmaRangeReduced,
                       .C = 2^(seq(-4, 6)))

SVMtrain <- train(traindata[,1:21],
                  y = traindata$trainClasses,
                  method = "svmRadial",
                  metric = "ROC",
                  tuneGrid = svmGrid,
                  fit = FALSE,
                  trControl = ctrl)
SVMtrain
plot(SVMtrain,  main = "Support Vector Machine")
head(SVMtrain$pred)
length(SVMtrain$pred[,1])

#confusion matrix - training data
confusionMatrix(data = SVMtrain$pred$pred,
                reference = SVMtrain$pred$obs,
                positive = "Success")

#ROC curve - training data
trainROC.SVM <- roc(response = SVMtrain$pred$obs,
                    predictor = SVMtrain$pred$Success,
                    levels = rev(levels(SVMtrain$pred$obs)))
plot(trainROC.SVM, legacy.axes = TRUE, main = "Training Data ROC Curve: SVM")
auc(trainROC.SVM)

#testing data
predict.SVM <- predict(SVMtrain, testdata[,1:21], type="prob")
testPredictors$SVMprob <- predict.SVM[,"Success"]
testPredictors$SVM.class <- predict(SVMtrain, testdata[,1:21])
testPredictors$obs <- as.factor(testClasses)

#confusion matrix - testing data
confusionMatrix(data = testPredictors$SVM.class,
                reference = testPredictors$obs,
                positive = "Success")

#ROC Curve - testing data
testROC.SVM <- roc(response = testPredictors$obs,
                   predictor = testPredictors$SVMprob,
                   levels = rev(levels(testPredictors$obs)))
plot(testROC.SVM, legacy.axes = TRUE, main = "Testing Data ROC Curve: SVM")
auc(testROC.SVM)

######################
# K-Nearest Neighbors
######################
set.seed(200)
Knntrain <- train(traindata[,1:21],
                  y = traindata$trainClasses,
                  method = "knn",
                  metric = "ROC",
                  tuneGrid = data.frame(.k = 1:400),
                  trControl = ctrl)
Knntrain
plot(Knntrain, main = "K-NN")
head(Knntrain$pred)
length(Knntrain$pred[,1])

#confusion matrix - training data
confusionMatrix(data = Knntrain$pred$pred,
                reference = Knntrain$pred$obs,
                positive = "Success")

#ROC curve - training data
trainROC.Knn <- roc(response = Knntrain$pred$obs,
                    predictor = Knntrain$pred$Success,
                    levels = rev(levels(Knntrain$pred$obs)))
plot(trainROC.Knn, legacy.axes = TRUE, main = "Training Data ROC Curve: Knn")
auc(trainROC.Knn)

#testing data
predict.Knn <- predict(Knntrain, testdata[,1:21], type="prob")
testPredictors$Knnprob <- predict.Knn[,"Success"]
testPredictors$Knn.class <- predict(Knntrain, testdata[,1:21])
testPredictors$obs <- as.factor(testClasses)

#confusion matrix - testing data
confusionMatrix(data = testPredictors$Knn.class,
                reference = testPredictors$obs,
                positive = "Success")

#ROC Curve - testing data
testROC.Knn <- roc(response = testPredictors$obs,
                   predictor = testPredictors$Knnprob,
                   levels = rev(levels(testPredictors$obs)))
plot(testROC.Knn, legacy.axes = TRUE, main = "Testing Data ROC Curve: Knn")
auc(testROC.Knn)

###############
# Naive Bayes
###############
set.seed(200)
nBayestrain <- train(traindata[,1:21],
                     y = traindata$trainClasses,
                     method = "nb",
                     metric = "ROC",
                     tuneGrid = data.frame(.fL = 2, .usekernel = TRUE, .adjust = TRUE),
                     trControl = ctrl)
nBayestrain
head(nBayestrain$pred)
length(nBayestrain$pred[,1])

#confusion matrix - training data
confusionMatrix(data = nBayestrain$pred$pred,
                reference = nBayestrain$pred$obs,
                positive = "Success")

#ROC curve - training data
trainROC.nBayes <- roc(response = nBayestrain$pred$obs,
                    predictor = nBayestrain$pred$Success,
                    levels = rev(levels(nBayestrain$pred$obs)))
plot(trainROC.nBayes, legacy.axes = TRUE, main = "Training Data ROC Curve: Naive Bayes")
auc(trainROC.nBayes)

#testing data
predict.nBayes <- predict(nBayestrain, testdata[,1:15], type="prob")
testPredictors$nBayesprob <- predict.nBayes[,"Success"]
testPredictors$nBayes.class <- predict(nBayestrain, testdata[,1:15])
testPredictors$obs <- as.factor(testClasses)

#confusion matrix - testing data
confusionMatrix(data = testPredictors$nBayes.class,
                reference = testPredictors$obs,
                positive = "Success")

#ROC Curve - testing data
testROC.nBayes <- roc(response = testPredictors$obs,
                   predictor = testPredictors$nBayesprob,
                   levels = rev(levels(testPredictors$obs)))
plot(testROC.nBayes, legacy.axes = TRUE, main = "Testing Data ROC Curve: Naive Bayes")
auc(testROC.nBayes)
