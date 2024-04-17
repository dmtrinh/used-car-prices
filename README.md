# What drives the price of a car?

## Context
 
Our goal is to understand what factors make a car more or less expensive.  Based on our analysis, we need to provide clear recommendations to the client -- a used car dealership -- as to what consumers value in a used car.

## Data

The data comes to us from [Kaggle](https://www.kaggle.com/).  The original dataset contained information on 3 million used cars.  The dataset we'll be working with contains information on 426K cars to ensure speed of processing.

# Approach

We will use the CRISP-DM framework to guide our approach.
![CRISP-DM Framework](./images/crisp.png)

## Business Understanding

From a business perspective, we are tasked with identifying key drivers for used car prices.  We will employ an essemble of techniques to determine feature importance and enable predictive modeling:
* Exploratory data analysis (EDA) to understand distribution of each feature and also their relationship to price
* Data preparation activities to filter out missing data, impute missing values, and engineer new features.
* Perform principal component analysis and clustering
* Build a model to help predict sales prices based on various attributes of a car

## Exploratory Data Analysis

