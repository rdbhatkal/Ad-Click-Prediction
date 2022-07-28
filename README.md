# Ad-Click-Prediction
## Logistic Regression

**Reeva Bhatkal** 

## Business problem:

A marketing agency wants to understand the features of a user who are likely to click on an advertisement. Using a machine learning algorithm we will develop a prediction model, that predicts if a particular user will click on an advertisement or not. 


## Data:
The following dataset was obtained from Kaggle.
Kaggle: https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad/code
The dataset contains 1000 observations with 10 features.

Here is the Data Dictionary for this dataset:



| Variable Name               | Description |
| ----------------------------|:-------------:|
| Daily Time Spent on a Site  | Time spent by the user on a site in minutes    |
| Age                         | Customer's age in terms of years     |
| Area Income                 | Average income of geographical area of consumer   |
| Daily Internet Usage        | Average minutes in a day consumer is on the internet     |
| Ad Topic Line               | Headline of the advertisement    |
| City                        | City of the consumer     |
| Male                        | Whether or not a consumer was male     |
| Country                     | Country of the consumer     |
| Timestamp                   | Time at which user clicked on an Ad or the closed window     |
| Clicked on Ad               | 0 or 1 is indicated clicking on an Ad     |


*Note: Please note that the data may have missing values. If missing values exist, we will be required to treat missing values accordingly.

## Methodology
1. Use pandas library to perform data cleaning: 
    1. Check for duplicate values
    2. Identifying and handling missing values 
    3. Look for outliers in the dataset
2. Adding more features to our dataset using Feature Engineering techniques
3. Create Exploratory Data Analysis on the data using pandas.
4. Create Explanatory visualization of feature distributions and correlation using matplotlib, seaborn and pandas
5. Build logistic regression models on selected features and target variable  
    1. Identify types of features (Numeric, Nominal)
    1. Transform each type of feature for machine learning
    1. Use StandardScalar to scale our data.
    1. Use OneHotEncoder() to one-hot encode nominal categorical features.
    1. Use ColumnTransformer to perform different strategies on different columns types
    1. Combine pipelines and column transformers to perform multiple transformations on different subsets of data.
6. Evaluate the logistic regression model using accuracy, recall and precision scores in Python.
7. Choose the most important metric for the analysis



## Observations


### AD CLICKS BASED ON DAILY TIME SPENT ON SITE
<img src = "Images/Obv 1.jpg" >


From the above plot, we can observe that even though people are spending more time on the internet they are not clicking more ads

Additionaly:

* An increase in 'Daily Time Spent on the Site' doesnt imply that the person is more likely to click on Ads.

* We see that for people who spend around 55 mins on the site daily are more likely to click on Ads.

* Women tend to spend more time on the internet than males

* Women also tend to click on Ads slightly more than Men


### TBD 


## Model

We build a Random Forest, KNN Classifier and Logistic Regression model to understand which is more reliable for prediction of the Clicked on Ads variable.

### Random Forest
The default Random Forest gives the following metrics: 
* Accuracy of Train set 1.0
* Accuracy of Test set 0.94

We try to improve our model by tuning the hyperparameters.
By Setting our Random Forest Classifier model with n_estimator = 50 and max_depth = 4  gives the following metrics: 
* Training accuracy: 0.976
* Test accuracy: 0.944

We can see that after tuning our hyperparameters, our accuracy on the test set has increased from 94% to 94.4%

<img src = "Images/RFMatrix.jpg" >

The confusion matrix shows that this model is slightly better at predicting users who clicked on Ads vs users who did not click on Ads.

In order to get the full picture of how our model is performing we can take a look at the clasification report

<img src = "Images/RFObs.JPG" >

We can see that the Random Forest Classifier model gives us an overall accuracy of 94%

### KNN Classifier
The default KNN model gives the following metrics: 
* Accuracy of Train set 0.969
* Accuracy of Test set 0.924

We try to improve our model by tuning the hyperparameters.
By Setting our KNN Classifier model with n_neighbours = 3 gives the following metrics: 
* Training accuracy: 0.973
* Test accuracy: 0.94

We can see that after tuning our hyperparameters, our accuracy on the test set has increased from 92.4% to 94%

<img src = "Images/KNNMatrix.jpg" >

The confusion matrix shows that this model is very good at predicting users who did not click on Ads vs users who clicked on Ads.

In order to get the full picture of how our model is performing we can take a look at the clasification report

<img src = "Images/KNNObs.JPG" >

We can see that the KNN Classifier model gives us an overall accuracy of 94%

### Logistic Regression
The default logistic regression model gives the following metrics: 
* Accuracy of Train set 0.984
* Accuracy of Test set 0.952

We try to improve our model by tuning the hyperparameters.
Next we apply L1 and L2 penalty and liblinear solver to see if we can improve our logistic regression model. We get the below metrics for L1 and L2 penalty 

* Accuracy of Train set with L1 penalty 0.9773333333333334
* Accuracy of Test set with L1 penalty 0.956

* Accuracy of Train set with L2 penalty 0.984
* Accuracy of Test set with L2 penalty 0.952

By Setting our logistic regression model with solver='liblinear', penalty='l1' and C=1 gives us an accuracy of 95.6% on our test set
By Setting our logistic regression model with solver='liblinear', penalty='l2' and C=1 gives us an accuracy of 95.2% on our test set

Both of these are similar to our default model. 
There is a very slight improvement to our model with 'l1' penalty so we use this model for our predictions 

<img src = "Images/LRMatrix.jpg" >

The confusion matrix shows that this model is slightly better at predicting users who did not click on Ads vs users who clicked on Ads.
In order to get the full picture of how our model is performing we can take a look at the classification report

<img src = "Images/LRObs.JPG" >

We can see that the Logistic Regression model gives us an overall accuracy of 96%


## Recommendations:




## Limitations & Next Steps

This dataset currently has very few datapoints for location, so we should work on collecting more data for specific countries and cities. Additionally, since the dataset has location and timestamp data, we can possibly create more powerful models using these features. 
With marketing over the internet taking over traditional marketing, a lot of companies prefer to advertise their products on the web. This concept of ad click prediction will allow companies to maximize their revenue while offering targeted marketing to customers 

