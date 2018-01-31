# -*- coding: utf-8 -*-
"""
@author: jason
"""

#importing dataframes and graphing libraries for work
#re is for parsing out regular expressions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

titanicData = pd.read_csv('train.csv')
titanicData.info()
print (titanicData.head(10))
plt.show()
plt.figure(0)
sns.heatmap(titanicData.isnull())


#Examine how gender and age affect survival rates
h = sns.FacetGrid(titanicData, col="Survived",  row="Sex")
h.map(plt.hist, "Age",edgecolor='black',bins=15)
plt.show()
plt.figure(1)

#Look at survival rates by age
ageCats = pd.cut(titanicData['Age'],[0,13,18,30,45,60,100])
print (titanicData.groupby(ageCats)['Survived'].mean())

#look at survival rates by age and gender
titanicData['ageBins'] = ageCats
ageGenderPivot = pd.pivot_table(data=titanicData, values='Survived', index='ageBins',columns='Sex',aggfunc = np.mean)
ageGenderPivot.plot(kind='bar')

plt.show()
plt.figure(2)

#Look at how SibSp, Pclass, Embarked, and Parch affect survival rates
print(titanicData[['SibSp','Survived']].groupby('SibSp').agg(['mean','count'])['Survived'])
print(titanicData[['Pclass','Survived']].groupby('Pclass').agg(['mean','count'])['Survived'])
print(titanicData[['Embarked','Survived']].groupby('Embarked').agg(['mean','count'])['Survived'])
print(titanicData[['Parch','Survived']].groupby('Parch').agg(['mean','count'])['Survived'])

#See what story embarkation and gender together tell us
h = sns.FacetGrid(titanicData, col='Embarked')
h.map(sns.pointplot,'Pclass','Survived','Sex',palette='colorblind',order=[1,2,3],hue_order=['male','female'])
h.add_legend()

plt.show()
plt.figure(3)
print(titanicData[titanicData['Embarked']=='C'].groupby('Sex').agg(['mean','count'])['Survived'])


#Investigate the distribution of fares
plt.figure(4,figsize=(12,10))
plt.hist(titanicData['Fare'],ec='black',bins=20)
plt.show()

plt.figure(5)
titanicData.groupby(pd.cut(titanicData['Fare'],bins=15)).Survived.agg(['mean']).plot(kind='bar')
plt.show()


#Presumably the higher class, the higher the average fare
#plt.figure(6)
#i = sns.pointplot(x='Pclass',y='Fare',data=titanicData,palette='colorblind')
#plt.show()

#Any insights to be gained from gender and PClass
#GRAPH MORE INSIGHTS HERE AGAINST GENDER
pivot_class = pd.pivot_table(index='Pclass',values = 'Survived',columns='Sex',data=titanicData,aggfunc=np.mean)
pivot_SibSp = pd.pivot_table(index='SibSp',values = 'Survived',columns='Sex',data=titanicData,aggfunc=np.mean)
pivot_Parch = pd.pivot_table(index='Parch',values = 'Survived',columns='Sex',data=titanicData,aggfunc=np.mean)

print (pivot_class)
print (pivot_SibSp)
print (pivot_Parch)


def getSuffix(expression):
    m = re.match('(.*?)\s*?(\w+)\.(.*)', expression)
    return m.group(2)

#Can we gain some insights from Suffixes?
titanicData['Suffix'] = titanicData['Name'].apply(getSuffix)
print(titanicData[['Survived','Suffix']].groupby(['Suffix'],as_index=False).agg(['mean','count']).xs('Survived',axis=1,drop_level=True).sort_values(by="mean",ascending=False))

#Can we deduce an age for null values based on class and Parch?
removeNull = (titanicData.drop(['Cabin'],axis=1)).dropna()
print(removeNull[['Age','Pclass','Sex']].groupby(['Pclass','Sex']).agg(['mean','count'])['Age'])


