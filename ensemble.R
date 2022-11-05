#_______________________________________________________________________________

## PRE-PROCESSING DATA
#_______________________________________________________________________________
#install.packages("tidyverse")
#install.packages("tidymodels")
library(tidyverse)
library(tidymodels)

#import the dataset
data <- read_delim("bank.csv",  delim = ";", escape_double = FALSE, trim_ws = TRUE)
bank <- data

#remove duration column to be practical
bank <- subset(bank, select = -duration)

#remove duplicate rows -> no duplicate found
duplicate <- subset(bank, duplicated(bank))
duplicate

##treating outliers
#balance -> replace high value outliers
high <- quantile(bank$balance)[4] + 1.5*IQR(bank$balance) #get 95% percentile value
for (index in c(1:length(bank$balance))) {
  bank$balance[index] <- ifelse(bank$balance[index] > high, high, bank$balance[index])
} #replace high value outliers with the 95% percentile value for balance

#campaign -> replace high value outliers
high <- quantile(bank$campaign)[4] + 1.5*IQR(bank$campaign) #get 95% percentile value
for (index in c(1:length(bank$campaign))) {
  bank$campaign[index] <- ifelse(bank$campaign[index] > high, high, bank$campaign[index])
} #replace high value outliers with the 95% percentile value for campaign


##transforming values
#one-hot encoding
bank_ohe <- recipe(y ~ ., data = bank) %>% #start create a recipe for pre-processing data
  step_range(all_numeric()) %>% #normalise the numeric variables into same range from 0-1
  step_dummy(all_nominal_predictors()) %>% #create dummy encoding for categorical variables
  prep() #run the recipe
bank <- juice(bank_ohe) #extract the data set from recipe to bank
bank <- bank %>% relocate(y, .after = last_col()) #move y outcome column to the last

glimpse(bank) #check structure of bank after performing one-hot encoding

##assign factors for outcome variables
bank$y <- factor(bank$y)

##After feature selection, we get 6 significant variables
selected_vars <- c("poutcome_success", "month_oct", "contact_unknown", "day",  
                   "month_mar", "pdays")
bank <- bank %>% select(all_of(selected_vars), y)

#_______________________________________________________________________________

## SAMPLING DATA
#_______________________________________________________________________________

#install packages 
#install.packages("caret")
library(caret)

#splitting into training and testing data set
set.seed(444) #reproducibility
#Create data partition with stratified sampling 
indexes <- createDataPartition(bank$y, times = 1, p = 0.7, list = FALSE)
bank.train <- bank[indexes,] 
bank.test  <- bank[-indexes,]

#Dimensions 
dim(bank.train)                
dim(bank.test)

#Check for proportion of labels in both and training and testing split
prop.table(table(bank.train$y))
prop.table(table(bank.test$y))

#_______________________________________________________________________________

## STACKING ENSEMBLE MODEL
#_______________________________________________________________________________


# Load library - caretEmsemble is used to build stacking ensemble model
# Install.packages('caretEnsemble')
library(caretEnsemble)

#### TRAIN BASE MODELS
# Setting control parameter to train base models with 10-fold cross 
# validation repeated 3 times and 30 resampling folds for every model.
ensembleControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 3,
                           index = createFolds(bank.train$y,30),
                           savePredictions= 'all',
                           classProbs = TRUE)

# We use random forest and elastic net regression for the base models
algorithmList <- c('rf', 'glmnet')

# Train the base models with the control parameters above
set.seed(444)
models <- caretList(y~., data=bank.train, trControl=ensembleControl, 
                    methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results) #plot the results of two base models 

# Check correlations between
# the base models' results
modelCor(results)
splom(results) # plot the resamples matrix of base models' results

### TRAIN META MODEL USING THE BASE MODELS' PREDICTIONS
# Setting control parameter to train model using caret with 10-fold
# cross validation repeated 3 times
stackControl <- trainControl(method = "repeatedcv", 
                             number = 10, 
                             repeats = 3,
                             savePredictions= 'final')

# Stack using logistic regression - glm family binomial
# Train with the base models' results
set.seed(444)
stack.glm <- caretStack(models, method="glm", metric="Accuracy",
                       trControl=stackControl, family = 'binomial')
summary(stack.glm)

### VALIDATE MODEL USING TEST SET
pred <- predict(stack.glm,bank.test)
# Confusion matrix for the stacking ensemble model
confusionMatrix(pred,bank.test$y)

## Extract the test data y to build the confusion matrix
bank.confusion <- table(pred,bank.test$y)
print(bank.confusion)

## Calculate accuracy, precision, recall, F1 using formulas
# Accuracy
bank.accuracy <- sum(diag(bank.confusion)) / sum(bank.confusion)
print(bank.accuracy)

# Precision for yes class

bank.precision.yes <- bank.confusion[2,2] / sum(bank.confusion[2,])
print(bank.precision.yes)

# Recall for yes class

bank.recall.yes <- bank.confusion[2,2] / sum(bank.confusion[,2])
print(bank.recall.yes)

# F1-measure
bank.f1 <- 2 * bank.precision.yes * bank.recall.yes / 
  (bank.precision.yes + bank.recall.yes)
print(bank.f1)







