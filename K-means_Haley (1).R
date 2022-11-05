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

#K-means Clustering Model
install.packages('factoextra')
library("factoextra")
library("cluster")

bank_y = ifelse(bank$y == 'yes', 1, 0)
bank = cbind(bank, bank_y)

#remove non-numeric columns as k-means clusters numerical vectors only
bank1 <- bank[, -c(7)]

bank1 <- na.omit(bank1) #listwise deletion of missing 

bank1 <- scale(bank1) #standardize variables only works for fully numerical datasets

#k=2
kmeans2 <- kmeans(bank1, centers = 2, nstart = 25)
kmeans2
fviz_cluster(kmeans2, data = bank1)

kmeans3 <- kmeans(bank1, centers = 3, nstart = 25)
kmeans4 <- kmeans(bank1, centers = 4, nstart = 25)
kmeans5 <- kmeans(bank1, centers = 5, nstart = 25)

#plot to compare
p1 <- fviz_cluster(kmeans2, geom = "point", data = bank1) + ggtitle("k = 2")
p2 <- fviz_cluster(kmeans3, geom = "point", data = bank1) + ggtitle("k = 3")
p3 <- fviz_cluster(kmeans4, geom = "point", data = bank1) + ggtitle("k = 4")
p4 <- fviz_cluster(kmeans5, geom = "point", data = bank1) + ggtitle("k = 5")

library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)

#Determine number of k-means clusters
set.seed(444)
wss <- (nrow(bank1)-1)*sum(apply(bank1,2,var))

for (i in 2:15)
  wss[i] <- sum(kmeans(bank1, centers=i)$withinss)

plot(1:15, wss, type="b", xlab="Number of clusters",
     ylab="Within groups sum of squares")

#compute k-means with k = 4
set.seed(444)
k4 <- kmeans(bank1, 4, nstart = 25)
#print the results
print(k4)

#plot results of final k-means model

fviz_cluster(k4, data = bank1, palette = "jco", ggtheme = theme_minimal())

#find means of each cluster
aggregate(bank1, by=list(k4$cluster),FUN=mean)

#add cluster assignment to original data
final_data <- cbind(bank1, cluster = k4$cluster)


table(bank$bank_y, k3$cluster)
#view final data
head(final_data)




