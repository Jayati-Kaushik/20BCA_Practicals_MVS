# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 21:34:47 2022

@author: KAIF
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
mydata=pd.read_csv("C:/Users/ADMIN/Documents/football.csv")
mydata.info()
print(mydata.head())
print(mydata.describe())
sns.countplot(x='Primary Skill',data=mydata)
sns.countplot(x='Secondary Skill',data=mydata)
sns.countplot(x='Cristiano Ronaldo',data=mydata)
sns.countplot(x='Lionel Messi',data=mydata)
sns.countplot(x='Neymar',data=mydata)
sns.scatterplot('Primary Skill','Secondary Skill',data=mydata)
sns.boxplot(y="Lionel Messi", x="Cristiano Ronaldo",data=mydata)
sns.displot(mydata["Lionel Messi"])
sns.jointplot(mydata["Lionel Messi"], mydata["Neymar"])
sns.pointplot(mydata['Lionel Messi'],mydata['Neymar'],hue=['Cristiano Ronaldo'])
plt.show()