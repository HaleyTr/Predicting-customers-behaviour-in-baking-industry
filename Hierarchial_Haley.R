library(tidyverse)
library(tidymodels)
#import the dataset
data <- read_delim("bank.csv",  delim = ";", escape_double = FALSE, trim_ws = TRUE)
bank <- mydata

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

## -> move to sampling data

#install packages 
install.packages("caret")
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

#K-means Model
install.packages('factoextra')
library("factoextra")

#in here we remove non-numeric columns as k-means clusters numerical vectors only
banktrain1 <- mydata[, -c(2,3,4,5,7,8,9,11,15,16,17)]

banktrain1 <- na.omit(banktrain1) #listwise deletion of missing 
banktrain1 <- scale(banktrain1) #standardize variables only works for fully numerical datasets

banktrain1

#Determine number of k-means clusters
wss <- (nrow(banktrain1)-1)*sum(apply(banktrain1,2,var))

for (i in 2:20)
  wss[i] <- sum(kmeans(banktrain1, centers=i)$withinss)

plot(1:20, wss, type="b", xlab="Number of clusters",
     ylab="Within groups sum of squares")

#compute k-means with k = 7
kmeans7 <- kmeans(banktrain1, 7, nstart = 25)
#print the results
print(kmeans7)

fviz_cluster(kmeans7, data = banktrain1, palette = "jco", ggtheme = theme_minimal())

#find means of each cluster
aggregate(banktrain1, by=list(kmeans7$cluster),FUN=mean)

final_data <- cbind(banktrain1, cluster = kmeans7$cluster)
head(final_data)
