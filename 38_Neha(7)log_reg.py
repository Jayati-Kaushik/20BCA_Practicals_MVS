
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
mydata=pd.read_csv("C:/Users/Santosh/Desktop/Neha/salary_data.csv")
print(mydata.shape)
print(mydata.describe())
print(mydata.dtypes)
print(mydata.value_counts())
# Do we have missing values?
print('Number of Missing Values:', mydata.isnull().sum().sum())
left = mydata[ mydata['left'] == 1]
print(left.shape)
retained = mydata[mydata['left'] == 0]
print(retained.shape)
data=pd.DataFrame(mydata.groupby('left').count())
#sns.countplot(x="left",data=mydata)
#sns.countplot(x="left",hue="salary",data=mydata)
#corr_ht=mydata.corr()
#print(corr_ht)
#sns.heatmap(corr_ht)
sns.countplot(x="salary",hue="left",data=mydata)
sns.countplot(x="Department",hue="left",data=mydata)
sns.countplot(x="satisfaction_level",hue="left",data=mydata)
sns.countplot(x="promotion_last_5years",hue="left",data=mydata)
newd=mydata[["left","satisfaction_level","salary","promotion_last_5years"]]

X_train,X_test, y_train, y_test =train_test_split(newd[["left","satisfaction_level","salary","promotion_last_5years"]],mydata.left,test_size=0.3, stratify=newd['left'])
model=LogisticRegression()
model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test, y_test)
# Create a scatter plot
#scatter(x, y, c=y, cmap='rainbow')
#plt.title('Scatter Plot of Logistic Regression')
#plt.show()