
#import typical data analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re

#import sklearn packages
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

#read in data and see head of data
train = pd.read_csv('train.csv')

#Get average age by class and sex to estimate age
estimateAge = (train.drop(['Cabin'],axis=1)).dropna().groupby(['Pclass','Sex']).agg(['mean'])['Age']

#Fill out age with means
def fillAge(age):
    if pd.isna(age[0]):
        return estimateAge.loc[age[1]].loc[age[2]][0]
    else:
        return age[0]

#Get suffixes
def getSuffix(expression):
    m = re.match('(.*?)\s*?(\w+)\.(.*)', expression)
    return m.group(2)

#Clean up data. Change some predictors to categorical variables
#Create dummy variables
def cleanUpData(df):
    df['Age'] = df[['Age','Pclass','Sex']].apply(fillAge,axis=1)
    #Remove unused columns 
    df.drop(['Ticket','Cabin','PassengerId'],inplace=True,axis=1)

    #Get dummy variables
    df = pd.get_dummies(df,columns=['Embarked','Sex','Pclass'],drop_first=True)

    #Break up columns into previously discussed categories
    df['Fare'] = df['Fare'].apply(lambda x: 'Low' if x <= 30 else ('Medium' if x <= 100 else 'High'))
    df['Parch'] = df['Parch'].apply(lambda x: 'NoChildPar' if x == 0 else ('OneChildPar' if x ==1 else 'HasChildPar'))
    df['SibSp'] = df['SibSp'].apply(lambda x: 'OnlyChild' if x == 0 else ('OneSib' if x in [1,2] else 'MulSib'))
    df['Age'] = df['Age'].apply(lambda x: 'Elderly' if x > 45 else ('Kid' if x<=13 else ('Adult' if x <=30 else 'Elderly')))

    #Dummify above categories
    df = pd.get_dummies(df,columns=['Fare','Parch','SibSp','Age'],drop_first=True)
    
    #Add interaction effect
    df['MalexKid'] = df['Sex_male']*df['Age_Kid']

    #Dummify suffix
    df['Name'] = df['Name'].apply(getSuffix)
    df['Name'] = df['Name'].apply(lambda x: 'Mrs' if x in ['Mrs','Countess','Ms','Mme','Lady','Mlle'] else ('Crew' if x in ['Capt','Rev','Don','Jonkheer','Dr','Major','Col'] else 'Other'))
    df = pd.get_dummies(df,columns=['Name'],drop_first=True)
    return df

#clean up training and test data
train = cleanUpData(train)
test = pd.read_csv('test.csv')
passengerId = test['PassengerId']
test = cleanUpData(test)

trainY=train['Survived']
trainX = train[train.drop(['Survived'],axis=1).columns]

test = test[train.drop(['Survived'],axis=1).columns]

#Perform LogisticRegression
lr = LogisticRegression()
lr.fit(trainX,trainY)
print ('Correct Predictions %: ',(lr.predict(trainX)==trainY).agg('mean'))
print (classification_report(trainY,lr.predict(trainX)))


#Use 9-fold cross validation for nearest neighbors
kf = KFold(n_splits = 9)

#Use Elbow method to choose best parameters for nearest neighbors
error_rate=[]
for i in range(1,30):
    kn = KNeighborsClassifier(i)
    error = 0
    for trainSet,cvSet in kf.split(train):
        kn.fit((train.loc[trainSet]).drop(['Survived'],axis=1),(train.loc[trainSet])['Survived'])
        kpredict = kn.predict((train.loc[cvSet]).drop(['Survived'],axis=1))
        error = np.mean(kpredict != train.loc[cvSet]['Survived'])
        error +=error
    error_rate.append(error/9)
plt.plot(error_rate,ls='--',markersize=8,marker='o')

#17 seems like an optimal nearest neighbor
kn=KNeighborsClassifier(17)
kn.fit(trainX,trainY)
print ('Correct Predictions %: ',(kn.predict(trainX)==trainY).agg('mean'))
print (classification_report(trainY,kn.predict(trainX)))


#Predict for test
predictions = kn.predict(test) #lr.predict(test)
submission = pd.DataFrame()
submission['PassengerId'] = passengerId
submission['Survived'] = predictions

submission.to_csv('output.csv',index=False)
