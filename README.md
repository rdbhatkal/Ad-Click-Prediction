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
    1. Deleting duplicate values
    1. Identifying and handling missing values by imputing its mean
    1. Replacing the miscoded information 
2. Create Exploratory Data Analysis on the data using pandas.
3. Create Explanatory visualization of feature distributions and correlation using matplotlib, seaborn and pandas
4. Build regression models on selected features and target variable  
    1. Identify three types of features (Numeric, Ordinal, Nominal)
    1. Transform each type of feature for machine learning
    1. Use ordinal encoding for ordinal categorical features.
    1. Use OneHotEncoder() to one-hot encode nominal categorical features.
    1. Use ColumnTransformer to perform different strategies on different columns types
    1. Combine pipelines and column transformers to perform multiple transformations on different subsets of data.
5. Evaluate the logistic model using accuracy, recall and precision scores in Python.
6. Choose the most important metric for the analysis



## Observations


### AD CLICKS BASED ON DAILY TIME SPENT ON SITE



From the above plot, we can observe that even though people are spending more time on the internet they are not clicking more ads

Additionaly:

* An increase in 'Daily Time Spent on the Site' doesnt imply that the person is more likely to click on Ads.

* We see that for people who spend around 55 mins on the site daily are more likely to click on Ads.

* Women tend to spend more time on the internet than males

* Women also tend to click on Ads slightly more than Men



