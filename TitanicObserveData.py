# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 23:58:44 2018

@author: jason
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import re

#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
#from sklearn.metrics import classification_report

titanicData = pd.read_csv('train.csv')

#SibSp effect on survival rate
print (titanicData[['Survived','SibSp']].groupby(['SibSp'],as_index=False).mean())



print (titanicData[['Survived','Pclass']].groupby(['Pclass'],as_index=False).mean())

print(titanicData[['Survived','Embarked']].groupby(['Embarked'],as_index=False).mean())

print(titanicData[['Survived','Parch']].groupby(['Parch'],as_index=False).mean())


a = plt.figure(2)
h = sns.FacetGrid(titanicData, row='Embarked')
h.map(sns.pointplot,'Pclass','Survived','Sex',palette='colorblind')
h.add_legend()
plt.show()
plt.figure(1)
g = sns.FacetGrid(titanicData, col="Survived",  row="Sex")
g.map(plt.hist, "Age",edgecolor='black',bins=20)

print(titanicData[['Age','Survived']].groupby('Age').agg(['mean','count']).xs('Survived',axis=1,drop_level=True).iloc[0:50])
print(titanicData[['Age','Survived']].groupby('Age').agg(['mean','count']).xs('Survived',axis=1,drop_level=True).iloc[51:])

print(titanicData[['SibSp','Survived']].groupby('SibSp').agg(['mean','count']).xs('Survived',axis=1,drop_level=True))
print(titanicData[['Parch','Survived']].groupby('Parch').agg(['mean','count']).xs('Survived',axis=1,drop_level=True))


plt.show()

b = plt.figure(3)
i = sns.pointplot(x='Pclass',y='Fare',data=titanicData,palette='colorblind')
plt.show()

def getTitle(expression):
    m = re.match('(.*?)\s*?(\w+)\.(.*)', expression)
    return m.group(2)

def getTitleOrdinal(title):
    if title in ['Countess','Ms','Mme','Lady','Mlle','Mrs']:
        return 0
    elif title == 'Miss':
        return 1
    elif title in ['Sir','Master','Col','Major','Dr']:
        return 2
    elif title == ' Mr':
        return 3
    else:
        return 4

def SibSpOrdinal(SibSp):
    if SibSp ==0 or SibSp ==1:
        return 0
    elif SibSp == 0:
        return 1
    else:
        return 2
    
def getOrdinalFare(fare):
    if fare <= 10:
        return 0
    elif fare >10 and fare <= 20:
        return 1
    elif fare >20 and fare <=50:
        return 2
    else:
        return 3


titanicData['Title'] = titanicData['Name'].apply(getTitle)

print(titanicData[['Survived','Title']].groupby(['Title'],as_index=False).agg(['mean','count']).xs('Survived',axis=1,drop_level=True).sort_values(by="mean",ascending=False))

plt.figure(figsize=(20,15))
plt.hist(titanicData['Fare'],ec='black',bins=30)
plt.show()

print(titanicData.groupby(pd.cut(titanicData['Fare'],bins=50)).Survived.agg(['mean','count']))
    

lowClass = titanicData[['Survived','Sex','Pclass']]
print(lowClass[lowClass['Pclass']==1].groupby('Sex').agg(['mean','count']).xs('Survived',axis=1,drop_level=True))
print(lowClass[lowClass['Pclass']==2].groupby('Sex').agg(['mean','count']).xs('Survived',axis=1,drop_level=True))
print(lowClass[lowClass['Pclass']==3].groupby('Sex').agg(['mean','count']).xs('Survived',axis=1,drop_level=True))