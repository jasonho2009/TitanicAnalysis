# Titanic
Data Analysis on Kaggle Dataset Titanic

## Introduction
The RMS Titanic was a British passenger ship that hit an iceberg on its maiden voyage and sank in 1912. It was the largest passenger ship afloat at the time and its sinking claimed the lives of over 1500 people, making it one of the worst maritime accidents in history. 

Kaggle, one of the foremost websites dedicated to data science and machine learning, provides a dataset on the passengers of the Titanic including whether the passenger survived or not, his/her age, the passenger's gender, ticket price, etc. We will use the data that Kaggle provides us to attempt to predict whether a passenger survived or not and which factors are significant to survival. 

The investigation will be broken down into these five stages:
1. Defining the problem
2. Get the data
3. Data exploration
4. Model the data
5. Communicate Results

### Defining the Problem
The topic we would like to explore in this project is to see if we can predict the survival of the Titanic passengers based on the data we have been given. If we know the personal statistics of a passenger, can we reasonably predict if the passenger survived or not?

The second question is to investigate which factors have the most impact on whether the passenger survives or not. For example, is the main factor gender? Or does the passenger's wealth play a more important role?
### Get the Data
Usually this section would be much more involved. Thankfully in this case, the data is already provided for us by Kaggle: [Kaggle](https://www.kaggle.com/c/titanic/data)

The data provided by Kaggle is also structured. They provide us with both a training data set and a test dataset for which we can test our modeling on. 
The Kaggle data provides whether the passenger survived or not, as well as other characteristics of the passenger such as gender, parch (# of parents/children on board), sibsp (# of siblings and spouses on board), and cabin number.
### Data Exploration
Here we explore what is the data available to us. We start by using pandas functions .info() and .head()
![pandas.info()](Images/info.png)
![pandas.head()](Images/dataHead.png)


### Model the Data
### Communicate Results
