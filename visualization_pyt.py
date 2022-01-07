# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 19:11:14 2022

@author: tejas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
mydata=pd.read_csv("http://winterolympicsmedals.com/medals.csv")
#mydata.info()
#print(mydata.head())
#print(mydata.describe())
#sns.countplot(x='City',data=mydata)
#sns.countplot(x='Sport',data=mydata)
#sns.countplot(x='Discipline',data=mydata)
#sns.countplot(x='Event gender',data=mydata)
#sns.countplot(x='Medal',data=mydata)
#sns.countplot(x='NOC',data=mydata)
#sns.countplot(x='Year',data=mydata)
#sns.countplot(x='Event',data=mydata)
#sns.scatterplot('Sport','Discipline',data=mydata)
#sns.boxplot(y="Year",x="Event gender",data=mydata)
#sns.distplot(mydata["Year"])
#sns.jointplot(mydata["Year"],mydata["Sport"])
#sns.pointplot(mydata['Year'],mydata['Sport'],hue=mydata['Event gender'])
#plt.show()