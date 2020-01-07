import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import normaltest

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

categorical = []
numerical = []
transform = []

def normality_test(data):
    res = True
    stat, p = normaltest(data)
    alpha = 0.05
    if p > alpha:
    	res = True
    else:
    	res = False
    return res

df_train.drop('Id',inplace=True,axis=1)
convert = ['GarageYrBlt','YrSold','MoSold','YearBuilt','YearRemodAdd']
for f in convert:
    df_train[f].astype(str)
    df_test[f].astype(str)

for col in df_train.columns:
    if df_train[col].dtype == 'object':
        categorical.append(col)
    else:
        numerical.append(col)

for num in numerical:
    df_train[num].fillna(df_train[num].mean(),inplace=True)
numerical.remove('SalePrice')
for num2 in numerical:
    df_test[num2].fillna(df_test[num2].mean(),inplace=True)

for cat in categorical:
    df_train[cat].fillna('None',inplace=True)
    df_test[cat].fillna('None',inplace=True)

le = preprocessing.LabelEncoder()
for categories in categorical:
    df_train[categories] = le.fit_transform(df_train[categories])
    df_test[categories] = le.fit_transform(df_test[categories])
corrmat = df_train.corr()
cols = corrmat.nlargest(31, 'SalePrice')['SalePrice'].index
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
test_id = df_test['Id']

for col2 in df_train.columns:
    s = normality_test(df_train[col2])
    if s != True:

        df_train[col2] = np.log1p(df_train[col2])
    if col2 not in list(most_corr['Most Correlated Features']):
        df_train.drop(col2,inplace=True,axis=1)

for col3 in df_test.columns:
    s = normality_test(df_test[col3])
    if s != True:
        df_test[col3] = np.log1p(df_test[col3])
    if col3 not in list(most_corr['Most Correlated Features']):
        df_test.drop(col3,inplace=True,axis=1)

df_y_train = df_train['SalePrice']

df_train.drop('SalePrice',inplace=True,axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(df_train, df_y_train, test_size=0.25, random_state=42)
reg = LinearRegression()
reg = reg.fit(X_train,y_train)
print(reg.score(X_train, y_train))
y_pred = reg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

prediction = np.expm1(reg.predict(df_test))


submit = pd.DataFrame({"Id" : test_id,"SalePrice" : prediction})
submit.to_csv('Prediction.csv',index=False)
