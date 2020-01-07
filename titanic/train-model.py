import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#read csv file
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
#combine train and test data
df_all = pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

#check missing data
missing = train_data.isnull().sum()

#analyze missing data within Embarked features
#print(train_data[train_data['Embarked'].isnull()])
#find len of string data in ticket
df_all["Ticket Num"] = df_all["Ticket"].str.len()
#find median len of str data and max fare
grp = df_all.groupby(['Embarked']).median()['Ticket Num']
grp2 = df_all.groupby(['Embarked']).max()['Fare']

df_all['C'] = df_all['Ticket'].str[0]
grp3 = df_all.groupby(by=['Embarked','C'])['C'].count()

#update missing data on embarked
train_data["Embarked"].fillna('S',inplace=True)
train_data["Embarked"].replace('C',0,inplace=True)
train_data["Embarked"].replace('Q',1,inplace=True)
train_data["Embarked"].replace('S',2,inplace=True)
test_data["Embarked"].replace('C',0,inplace=True)
test_data["Embarked"].replace('Q',1,inplace=True)
test_data["Embarked"].replace('S',2,inplace=True)

#analyze missing data within cabin
train_data.loc[train_data['Cabin'].str[0] == 'A', 'Cabin'] = 0
train_data.loc[train_data['Cabin'].str[0] == 'B', 'Cabin'] = 1
train_data.loc[train_data['Cabin'].str[0] == 'C', 'Cabin'] = 2
train_data.loc[train_data['Cabin'].str[0] == 'D', 'Cabin'] = 3
train_data.loc[train_data['Cabin'].str[0] == 'E', 'Cabin'] = 4
train_data.loc[train_data['Cabin'].str[0] == 'F', 'Cabin'] = 5
train_data.loc[train_data['Cabin'].str[0] == 'G', 'Cabin'] = 6
train_data.loc[train_data['Cabin'].str[0] == 'T', 'Cabin'] = 7
train_data['Cabin'].fillna(8,inplace=True)

test_data.loc[test_data['Cabin'].str[0] == 'A', 'Cabin'] = 0
test_data.loc[test_data['Cabin'].str[0] == 'B', 'Cabin'] = 1
test_data.loc[test_data['Cabin'].str[0] == 'C', 'Cabin'] = 2
test_data.loc[test_data['Cabin'].str[0] == 'D', 'Cabin'] = 3
test_data.loc[test_data['Cabin'].str[0] == 'E', 'Cabin'] = 4
test_data.loc[test_data['Cabin'].str[0] == 'F', 'Cabin'] = 5
test_data.loc[test_data['Cabin'].str[0] == 'G', 'Cabin'] = 6
test_data['Cabin'].fillna(8,inplace=True)

#analyze missing data within age
df_all_corr = df_all.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']

# Filling the missing values in Age with the medians of Sex and Pclass groups
train_data['Age'] = train_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
test_data['Age'] = test_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
test_data['Fare'].fillna(med_fare,inplace=True)
train_data["Sex"].replace('male',0,inplace=True)
train_data["Sex"].replace('female',1,inplace=True)
test_data["Sex"].replace('male',0,inplace=True)
test_data["Sex"].replace('female',1,inplace=True)

train_data.drop(['Name','Ticket'], axis=1,inplace=True)
test_data.drop(['Name','Ticket'], axis=1,inplace=True)
y = train_data.pop('Survived')
model = RandomForestClassifier(max_depth = 30,n_estimators=100)
num_columns = list(train_data.dtypes[train_data.dtypes != 'object'].index)
model.fit(train_data[num_columns],y)
score = accuracy_score(y , model.predict(train_data[num_columns]),normalize=True)
pred = model.predict(test_data[num_columns])
pred.sum()
submit = pd.DataFrame({"PassengerId" : test_data["PassengerId"],"Survived" : pred})
submit.to_csv('Prediction.csv',index=False)
