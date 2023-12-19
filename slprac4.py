import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/content/drive/MyDrive/Salary_Data.csv')
df.head()
df.shape
X = df[['YearsExperience']].values
y = df['Salary'].values
plt.scatter(X,y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.intercept_
regressor.coef_
X_test[0]
regressor.predict([X_test[0]])
regressor.predict([[6.8]])
X_test
y_pred = regressor.predict(X_test)
y_pred
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
new_salary_pred = regressor.predict([[15]])
np.round(new_salary_pred,2)
y_test
X_test
pred = regressor.predict(X_test)
pred
for i in range(len(X_test)):
    print("Experience: ",X_test[i], '  predicted Salary',regressor.predict([X_test[i]]), ' Acual Salary', y_test[i] )
n = 9
MSE = (sum(y_test-y_pred)**2)/n
MSE
RMSE = np.sqrt((sum(y_test-y_pred)**2)/n)
RMSE
MAE = (sum(np.abs(y_test-y_pred)))/n
MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error
test_mse = mean_squared_error(y_test, y_pred)
print("the test MSE is: ",test_mse)
test_mae = mean_absolute_error(y_test, y_pred)
print("the test MAE is: ",test_mae)
test_rmse = np.sqrt(test_mse)
print("the test RMSE is: ",test_rmse)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
r2,regressor.score(X_test, y_test)
n=30
k=1
print(1 - ((1-r2)*(n-1)/(n-k-1)))
a = (1-r2)*(n-1)/(n-k-1)
adj_r2 = 1 - a
adj_r2
