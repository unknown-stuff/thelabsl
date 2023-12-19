import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
compdata = pd.read_csv('/content/drive/MyDrive/1000_Companies for univariant and multivariant (1) (1).csv')
compdata.head()
y = compdata['Profit'].values
y
X = np.column_stack([np.ones(len(compdata), dtype=np.float32),compdata['R&D Spend'].values])
X
x1 = compdata['R&D Spend'].values
x1
x2 = compdata['Administration'].values
x2
plt.scatter(x1, y)
plt.show()
plt.scatter(x2, y)
plt.show()
from statsmodels.regression.linear_model import OLS
regout = OLS(y,X).fit()
coeffs_ols = regout.params
print(f'coefficients : {coeffs_ols}')
plt.scatter(y, ols_preds)
plt.show()
regout.summary()
XX = np.column_stack([np.ones(len(compdata), dtype=np.float32),compdata['R&D Spend'].values,compdata['Administration'].values])
regout2 = OLS(y,XX).fit()
coeffs_ols2 = regout2.params
print(f'coefficients : {coeffs_ols2}')
ols_preds2= regout2.predict()
plt.scatter(y, ols_preds2)
plt.show()
regout2.summary()
