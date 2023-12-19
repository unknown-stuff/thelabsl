import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
file_path = '/content/drive/MyDrive/Ex1.xlsx'
df = pd.read_excel(file_path)
print(df)
y = df['Sales']
X = df[['Advertisement']]
plt.scatter(X,y,marker ='o')
lr = LinearRegression()
lr.fit(X,y)
lr.coef_
lr.intercept_
lr.predict([[200]])
lr.intercept_+200*lr.coef_
y_pred = lr.predict(X)
y_pred
plt.scatter(X,y)
plt.plot(X,y_pred,color = 'r')
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y,y_pred)
mse
