import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
train = pd.read_csv("train.csv")
train.head()
train.count()
train[train['Sex'].str.match("female")].count()
train[train['Sex'].str.match("male")].count()
sns.countplot(x='Survived', hue='Pclass', data=train)
sns.countplot(x='Survived', hue='Sex', data=train)
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)
train.drop("Cabin",inplace=True,axis=1)
train.dropna(inplace=True)
pd.get_dummies(train["Sex"])
sex = pd.get_dummies(train["Sex"],drop_first=True)
embarked = pd.get_dummies(train["Embarked"],drop_first=True)
embarked = pd.get_dummies(train["Pclass"],drop_first=True)
#train = pd.concat([train,pclass,sex,embarked],axis=1)
train.drop(["PassengerId","Pclass","Name","Sex","Ticket","Embarked"],axis=1,inplace=True)
X = train.drop("Survived",axis=1)
y = train["Survived"]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
