import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('cbb.csv')

# print the correlation to the number of wins
print(df.corr()['W'].sort_values()[:-1])
df_ff = df[['EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD','W']]

# plot the dsitribution of each features, comment out if you don't want to plot it
fig, axes = plt.subplots(ncols=4, nrows=3)
for col, ax in zip(df_ff.columns, axes.flat):
    sns.distplot(df_ff[col], hist=False, ax=ax)
plt.tight_layout()
plt.show()

# descriptive stats
print(df_ff.describe())

# prepare the dataset
df_ff.drop('W', inplace=True, axis=1)
df_ff_y = df['W']

# train the model
X_train, X_test, y_train, y_test = train_test_split(df_ff, df_ff_y, test_size=0.25, random_state=21)
reg = LinearRegression()
reg = reg.fit(X_train,y_train)

# analyze the model
print('Intercept: ', reg.intercept_)
print('R^2 score: ',reg.score(X_train,y_train))
coeff_df = pd.DataFrame(reg.coef_, df_ff.columns, columns=['Coefficient'])
print(coeff_df)

# predict from the test set
y_pred = reg.predict(X_test)

# analyze the prediction
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
dfd = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'AbsDiff': abs(y_test-y_pred)})
dfd.sort_values(by=['AbsDiff'], inplace=True, ascending=True)
#print(dfd[:10])
#print(df[df.index == 1477])
