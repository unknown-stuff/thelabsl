import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
adv = "/content/drive/MyDrive/Exp_5_Data_Bankrupt_Firm.csv"
df = pd.read_csv(adv, header=0)
print("\nShape of Dataset =", df.shape)
print("\nFeature name of Dataset = \n", df.columns)
print("\nPrint data types of the Dataset = \n", df.dtypes)
print("\nPrint top records of the Dataset =\n", df.head())
print("\nPrint bottom rows of the Dataset =\n", df.tail())
print("\nPrint top 10 rows of the Dataset =\n", df.head(10))
print("\nPrint bottom 10 rows of the Dataset =\n", df.tail(10))
print(df.corr())
print("\n")
print(df.corr()[(df.corr()>0.05) & (df.corr()< 1)])
import seaborn as sns
sns.set()
df.hist(figsize=(10,7), color='blue')
import matplotlib.pyplot as plt
plt.show()
x = df.iloc[:,1:4]
print('\nSample data of independent variables x = \n', x.head())
y = df.iloc[:,0:1]
print('\nSample data of dependent variables y = \n', y.head())
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print('\nShape of training and test data')
print('\nTraining Data for x (x_train) =,', x_train.shape)
print('\nTest Data for x (x_test) = ', x_test.shape)
print('\nTraining Data for y (y_train) =,', y_train.shape)
print('\nTest Data for y (y_test) = ', y_test.shape)
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(solver='lbfgs')
model = logReg.fit(x_train, y_train.values.ravel())
y_pred = logReg.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
acc_round = accuracy.round(3)
pre_round = precision.round(3)
rec_round = recall.round(3)
print("Accuracy:", acc_round)
print("Precision:", pre_round)
print("Recall:", rec_round)
y_pred_proba = logReg.predict_proba(x_test)[::,1]
fpr, tpr,_ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
