# Predicting customers' behaviour in baking industry

## **Introduction**

With data-driven decision-making, the banking industry can gain valuable and logical insights to create competitive advantages. Given a data set of a Portuguese banking institution’s direct marketing campaigns, business analysts and data analysts can develop predictive models to classify customers, thus gaining useful information to optimise the marketing campaigns. This analysis aims to provide the banking institutions with a predictive model to identify the clients who would subscribe to the bank term deposit, using the Portuguese bank’s marketing campaign data set.

## **Data set**

The data set is related with direct marketing campaigns of the Portugues banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

## **Method**

Supervised learning models learn from labelled data to identify patterns in the train data set. The models can classify and predict the new input data set without labels in the future. With supervised models, the Portuguese bank can predict new customers’ responses. This analysis uses three supervised algorithms:

1. Random Forest

2. Elastic Net Regression

3. Stacked Generalisation Ensemble

Furthermore, unsupervised learning models learn from the unlabelled data set. Unsupervised learning can classify the data into new groups by discovering the underlying patterns instead of assigning labels to new input data sets like supervised learning models. Using these new groups, the bank can identify insights to predict customers' behaviours and responses based on their segmentation. Two clustering models used are:

1. k-Means Clustering

2. Hierarchical Clustering

## **Result**

We found that the successful outcome of the previous marketing campaign influences the customers’ decision to subscribe to this current campaign. The accuracy scores for the classification model:

1. Random Forest 89.45%

2. Elastic Net Regression 89.60%

3. Stacked Generalisation 89.68%

Also, the Hierarchical Clustering model is more optimal compared to the K-means Clustering model for our approach.
