import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

df = pd.read_excel('./dataset/criminality_dataset.csv')
df.head()

df.describe()

df = df.dropna(how='any', axis=0)

df.columns
df_new = df

df_dummies = pd.get_dummies(data=df_new, columns = ['criminality', 'behavior', 'factors'])
df_dummies.head(10)

df_dummies.dtypes

df_dummies = df_dummies.drop(1930)

df_dummies['Kilometer'] = pd.to_numeric(df_dummies['criminality'])

X = df_dummies.drop('factor', axis=1)
X = df_dummies.drop('motive', axis=1)
X

y = df_dummies.Price
y

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

linreg.fit(X,y)

from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

np.sqrt(metrics.mean_squared_error(y_test, y_pred))

y_pred = linreg.predict(X_test)
print(linreg.score(X_test, y_test)*100, '# Prediction accuracy')



