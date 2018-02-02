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
Here we explore what is the data available to us. We start by using pandas functions .info() and .head() to get below data

info()
![pandas.info()](Images/info.png)

head()
![pandas.head()](Images/dataHead.png)

From the above we can see the type of data we have. First we note that there are 891 entries. Moreover, we are missing some data. The data for "Cabin", we only have 204 of the 891 entries.
Seems that this is one column we can drop as we don't even have a quarter of the data.

We are also missing a good chunk of data from "Age". However, the vast majority of the entries are there so perhaps we can still draw some conclusions from this data still

![heatmap of null values](Images/Heatmap.png)
We make a heatmap of the null values of the data above. We can see that indeed we are missing most of the Cabin data and a lot of the Age data. 
Obviously the Cabin data can be very important to determining who survives. Unfortunately we are missing most of the data so the conclusions we can draw are limited.
We are missing some of the Age data. However, we may be able to extrapolate some of the data

#### Age & Gender

#####Survival by Age and Sex Facet Grid
![Survival by Age and Sex](Images/SurvivalByAgeAndSex.png)

#####Survival by Age
![Survival by Age](Images/SurvivalByAge.png)

#####Survival by Age and Sex Bar Graph
![Survival by Age and Sex](Images/SurvivalByAgeAndSexBarGraph.png)

* From the above we can clearly conclude that there is strong correlation between survival rates and gender. From the first graph, seems that survival is greater for female than male for almost all ages
* We notice a trend in the third graph. For females, seems that the survival rate rose by age
* On the other hand, the trend is less obvious for males. Initially, the survival rate is quite high for young boys. Then there is a huge drop for adult men that rises slightly until 45 when it drops again
* Seems that the survival rate for boys is quite high comparatively. This makes sense as on the Titanic, women and children were allowed on lifeboats first. We don't see the same trend for young girls
* The second chart shows survival by age. Again we see young children have the highest survival rate. The survival rate is a bit spotty for the rest of the ages before dropping for 60+

-Overall from this information we conclude a few on variables. The age and survival rate correlation is spotty in the middle. So we break up age into kid, adult, and elderly. 
-In addition, we add a variable called MalexKid. We notice an interaction effect for gender and age. Seems that boys survive at a much higher rate. We don't see the same effect for girls. So we add this interaction effect to account for this.

#####Survival by Various Metrics
![Survival by Various Metrics](Images/SurvivalRatesVariousMetrics.png)

*SibSp represents the # of siblings and spouse that are on the Titanic for each passenger. We notice that 

### Model the Data
### Communicate Results
