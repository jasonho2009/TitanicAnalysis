# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:45:50 2018

@author: jason
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

titanicData = pd.read_csv('train.csv')
print(titanicData.head())

def genderConvert(sex):
    if sex == 'male':
        return 1
    else:
        return 0
def getTicketNumber(ticket):
    if ticket == "LINE":
        return 0
    else:    
        print(ticket.split(" ")[-1])
        return int(ticket.split(" ")[-1])
titanicData.drop('Cabin',axis=1,inplace=True)
titanicData['Gender'] = titanicData['Sex'].apply(genderConvert)

titanicData.drop('Sex',axis=1,inplace=True)

newEmbarked = pd.get_dummies(titanicData['Embarked'],drop_first=True)
newClass = pd.get_dummies(titanicData['Pclass'],drop_first=True,prefix='Class')
print(newClass)
titanicData.drop(['PassengerId','Embarked','Pclass'],axis=1,inplace=True)
titanicData = pd.concat([titanicData,newEmbarked,newClass],axis=1)

def kid(age):
    if age < 18:
        return 1
    else:
        return 0
    
def elderly(age):
    if age >= 50:
        return 1
    else:
        return 0

def getTitle(expression):
    m = re.match('(.*?)\s*?(\w+)\.(.*)', expression)
    return m.group(2)

def getOrdinalTitle(title):
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

titanicData['title'] = titanicData['Name'].apply(getTitle)
titanicData['OrdinalTitle'] = titanicData['title'].apply(getOrdinalTitle)
titanicData['OrdinalSibSp'] = titanicData['SibSp'].apply(SibSpOrdinal)
titanicData['OrdinalFare'] = titanicData['Fare'].apply(getOrdinalFare)

lm = LinearRegression()
nanAge = titanicData[np.isfinite(titanicData['Age'])]
lm.fit(nanAge[['Gender','Class_2','Class_3','OrdinalFare','OrdinalSibSp']],nanAge['Age'])
predictedAge = lm.predict(nanAge[['Gender','Class_2','Class_3','OrdinalFare','OrdinalSibSp']])
print(metrics.r2_score(nanAge['Age'],predictedAge))

def newAge(cols):
    age = cols[0]
    gender = cols[1]
    class2 = cols[2]
    class3 = cols[3]
    fare = cols[4]
    sib = cols[5]
    if pd.isnull(age):
        return max(0,lm.predict(np.array([gender,class2,class3,fare,sib]).reshape((1,5)))[0])
    else:
        return age

titanicData['Age'] = titanicData[['Age','Gender','Class_2','Class_3','OrdinalFare','OrdinalSibSp']].apply(newAge,axis=1)
titanicData['kid'] = titanicData['Age'].apply(kid)
titanicData['elderly'] = titanicData['Age'].apply(elderly)


def getOrdinalFare(fare):
    if not type(fare) == int:
        return 1
    if fare <= 10:
        return 0
    elif fare >10 and fare <= 20:
        return 1
    elif fare >20 and fare <=50:
        return 2
    else:
        return 3


def SibSpOrdinal(SibSp):
    if SibSp ==0 or SibSp ==1:
        return 0
    elif SibSp == 0:
        return 1
    else:
        return 2


titanicData.drop(['Name', 'Ticket', 'title','SibSp','Fare'], axis=1, inplace=True)

print("titanicHead", titanicData.head())

lr = LogisticRegression()
lr.fit(titanicData[['Gender','Parch', 'Q', 'S','Class_2', 'Class_3', 'kid','elderly','OrdinalTitle','OrdinalSibSp','OrdinalFare']],titanicData[titanicData.columns[0]])
classifications = classification_report(titanicData['Survived'],lr.predict(titanicData[['Gender','Parch', 'Q', 'S','Class_2', 'Class_3', 'kid','elderly','OrdinalTitle','OrdinalSibSp','OrdinalFare']]))

print(classifications)

testData = pd.read_csv('test.csv')

newEmbarked = pd.get_dummies(testData['Embarked'],drop_first=True)
newClass = pd.get_dummies(testData['Pclass'],drop_first=True,prefix='Class')
testData = pd.concat([testData,newEmbarked,newClass],axis=1)
testData['Gender'] = testData['Sex'].apply(genderConvert)

testData['title'] = testData['Name'].apply(getTitle)
testData['OrdinalTitle'] = testData['title'].apply(getOrdinalTitle)
testData['OrdinalSibSp'] = testData['SibSp'].apply(SibSpOrdinal)
testData['OrdinalFare'] = testData['Fare'].apply(getOrdinalFare)
testData['Age'] = testData[['Age','Gender','Class_2','Class_3','OrdinalFare','OrdinalSibSp']].apply(newAge,axis=1)

testData.drop(['Name','Pclass','Sex','Ticket','Cabin','title','Fare'],axis=1,inplace=True)

testData['kid'] = testData['Age'].apply(kid)
testData['elderly'] = testData['Age'].apply(elderly)

TrueFalse = testData.isnull()
sns.heatmap(data = TrueFalse)
testData['Predictions'] = lr.predict(testData[['Gender','Parch', 'Q', 'S','Class_2', 'Class_3', 'kid','elderly','OrdinalTitle','OrdinalSibSp','OrdinalFare']])
testData[['PassengerId','Predictions']].to_csv('output.csv')
for i in lr.coef_:
    for j in i:
        print(j)
        
print(titanicData[['Gender','Survived']].groupby('Gender').agg(['mean','count']).xs('Survived',axis=1,drop_level=True))
