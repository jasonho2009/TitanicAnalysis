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

#sns.countplot(x='Sex',data=titanicData,hue='Survived')

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
titanicData.drop(['PassengerId','Name','Embarked','Pclass'],axis=1,inplace=True)
titanicData = pd.concat([titanicData,newEmbarked,newClass],axis=1)


lm = LinearRegression()
nanAge = titanicData[np.isfinite(titanicData['Age'])]
lm.fit(nanAge[['Gender','Class_2','Class_3','Fare','SibSp']],nanAge['Age'])
predictedAge = lm.predict(nanAge[['Gender','Class_2','Class_3','Fare','SibSp']])
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

def getTitle(expression):
    m = re.match('(.*?)\s*?(\w+)\.(.*)', expression)
    return m.group(2)

titanicData['Age'] = titanicData[['Age','Gender','Class_2','Class_3','Fare','SibSp']].apply(newAge,axis=1)
print("here")

lr = LogisticRegression()
lr.fit(titanicData[['Age', 'SibSp', 'Parch', 'Fare', 'Gender', 'Q', 'S','Class_2', 'Class_3']],titanicData[titanicData.columns[0]])
classifications = classification_report(titanicData['Survived'],lr.predict(titanicData[['Age', 'SibSp', 'Parch', 'Fare', 'Gender', 'Q', 'S','Class_2', 'Class_3']]))
testData = pd.read_csv('test.csv')
print(classifications)

newEmbarked = pd.get_dummies(testData['Embarked'],drop_first=True)
newClass = pd.get_dummies(testData['Pclass'],drop_first=True,prefix='Class')
testData = pd.concat([testData,newEmbarked,newClass],axis=1)
testData['Gender'] = testData['Sex'].apply(genderConvert)
testData['Age'] = testData[['Age','Gender','Class_2','Class_3','Fare','SibSp']].apply(newAge,axis=1)
testData.drop('Cabin',axis=1,inplace=True)
testData.dropna(inplace=True)
testData['Predictions'] = lr.predict(testData[['Age', 'SibSp', 'Parch', 'Fare', 'Gender', 'Q', 'S','Class_2', 'Class_3']])
testData.to_csv('output.csv')
print("params: ",lr.coef_)

