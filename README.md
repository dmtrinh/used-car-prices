# What drives the price of a car?

## Context
 
Our goal is to understand what factors make a car more or less expensive.  Based on our analysis, we need to provide clear recommendations to the client -- a used car dealership -- as to what consumers value in a used car.

## Data

The data comes to us from [Kaggle](https://www.kaggle.com/).  The original dataset contained information on 3 million used cars.  The dataset we'll be working with contains information on 426K cars to ensure speed of processing.

# Approach

We will use the CRISP-DM framework to guide our approach.
![CRISP-DM Framework](./images/crisp.png)

## Business Understanding

From a business perspective, we are tasked with identifying key drivers for used car prices.  We will employ a collection of techniques to determine feature importance and support building of a predictive model for used car sale prices:
* Exploratory data analysis (EDA) to understand distribution of each feature and also their relationship to price
* Data preparation activities to filter out missing data, impute missing values, and engineer new features that may be useful.
* Linear regression with regularization to build a model to predict sales prices based on various features of a car

## Exploratory Data Analysis
Upon initial examination of the dataset, we immediately noticed a majority of features is non-numerical.  Furthermore, some features have substantial amounts of missing values.

<img src="./images/00_Dataframe_info.png" width="300" />
<img src="./images/01_Initial_percent_missing_values.png" width="200" />

`price` and `odometer` features have very large ranges, suggesting some of the values may not be legitimate.

<img src="./images/03_Stats_for_numerical_features.png" width="400" />

Notice the long tail for distribution of prices.

<img src="./images/04_Distribution_of_prices.png" width="400" />

## Data Preparation and Feature Engineering
An extensive amount of preparation work was needed to get the data into a usable state for machine learning:
* Given the high percentage of missing values for `VIN` (>37%) and `size` (>70%), these columns were dropped.
* The features [`year`, `model`, `fuel`, `odometer`, `transmission`] all had <1.5% missing values.  The _rows_ with missing values in these were dropped.
* Imputed missing values for `manufacturer` based on text in `model`; e.g. 'Scion' and 'Genesis'.
* Imputed missing values for `condition` feature.  'good' was used to fill in missing values due to its overwhelming frequency in the dataset.
* `condition` is an ordinal categorical feature; the values were label encoded as follows:
  * 'salvage'=0
  * 'fair'=1
  * 'good'=2
  * 'excellent'=3
  * 'like new'=4
  * 'new'=5
* A new feature `clean_title` was created based on the values from `title_status`.  `clean_title` will have a value 1 if the `title_status` is 'clean', otherwise, 0 will be used as the value.
* Dropped features `model`, `state`, `region`, due to the sheer number of categorical values
* Dropped features `cylinders`, `drive`, `type`, and `paint_color` as they had over 20% missing values.  
* Dropped all rows where `price` was > $400,000.  `price` had 35 rows with non-sensical values such as 123456789, 777777, 999999, etc.
* Dropped all rows where `price` was <= $0.  There were 28,770 rows that fit this condition.
* Odometer also had non-sensical values such as 8888888.0, 9999999.0, etc.  We capped the odometer value at 400,000 miles; 807 rows with values larger than 400K were dropped.
* Added a feature for `age` of car; dropped `year` feature
* One-hot encode `fuel` and `transmission` features
* Added feature `manufacturer_LOO_encoding` by performing Leave-One-Out target encoding of `manufacturer` feature
* Added feature `manufacturer_frequency_encoding` by performing frequency encoding of `manufacturer` feature
* Dropped `manufacturer` feature

The cleanup left us with 369K samples.  Correlation matrix for the cleaned dataset:
<img src="./images/10_Correlation_matrix.png" width="800" />

Leaving aside `log_price` which has obvious and direct correlation to `price`, we can observe that both `odometer` and `age` have fairly strong negative correlation to `price`.  This logically makes sense since older and higher mileage vehicles typically are not worth as much.

## Building the Regression Models
The dataset was divided into two sets: 75% train and 25% test split.  We employed this ML pipeline:
<img src="./images/11_Pipeline.png" width="500" />

Since features are in different scales, we ran a handful of features through the StandardScaler.  To retain all features for prediction, Ridge regression was intentionally chosen.  Implementation-wise, we used RidgeCV for its built-in cross validation.

After fitting, we have:

<img src="./images/12_Coef_importance.png" width="600" />

<img src="./images/13_Actual_vs_Predicted.png" width="600" />

Mean absolute error (MAE) is easier for customers to understand so we used that metric for measuring our model performance.  Our model performed very well with less than a 1% difference for MAE between our training set and test set.

## Findings and Recommendations
Our analysis suggests:
* A clean title will contribute positively to a car's price
* Cars consuming diesel fuel and/or having a non-standard transmission may also command higher prices.  The limited availability of cars with these features (i.e. small number of samples in our dataset) might have led to elevated prices.
* Electric cars seem to be falling out of favor given the large negative coefficient.
* Older cars and/or ones with higher mileage tend to sell for less.

To maximize revenue, car dealerships should focus on obtaining inventory of newer, lower mileage vehicles with clean titles.
