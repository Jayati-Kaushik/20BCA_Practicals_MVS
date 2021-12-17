# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:50:31 2021

@author: Surya
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("insurance.csv")
data.head()
data.shape
data.info()
data.drop_duplicates()
data.shape
data.describe()
data['region'].value_counts()
data['sex'].value_counts()
sns.pairplot(data)
data.corr()
sns.heatmap(data.corr(),annot=True)
