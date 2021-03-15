import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


data = pd.read_csv("student-mat.csv")

np.set_printoptions(threshold=sys.maxsize)


x1 = data.iloc[:, :3]
x2 = data.iloc[:, 4:6]
x3 = data.iloc[:, 10:-3]
y = data.iloc[:, -3:]

x = pd.concat([x1, x2, x3], axis=1)
pd.set_option("display.max_columns", 50)

column_transform = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0, 1, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17])], remainder='passthrough')

x = np.array(column_transform.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3)

linear_regressor = LinearRegression()

linear_regressor.fit(x_train, y_train)

y_pred = linear_regressor.predict(x_test)

np.set_printoptions(precision=2)
print("Predicted results")
print(y_pred)
print("-----------------------------------")
print("Actual results")
print(y_test)
print("-----------------------------------")
print()

sum = 0
num = 0
for i in range(len(x_test)):
    for j in range(3):
        dif = y_pred[i][j] - y_test.iloc[i, j]
        sum += abs(dif)
        num += 1

print("Average error margin on this test run:")
print(sum/num)

# Visualising the Training set results
# plt.scatter(x_train, y_train, color='red')
plt.plot(x_test, y_test, color='blue')
plt.show()
