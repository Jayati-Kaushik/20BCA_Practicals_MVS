# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 22:14:13 2022

@author: Shahid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

os.chdir("C:/Users/Shahid Khan/Downloads/steam.csv")
st = pd.read_csv('steam.csv')
st.head()
print(st.head())
print(st.tail())
print(st.columns)
print(st.describe())
print(st.nunique())

#To check Null values
print(st.isnull().sum())
print(st.head())
print(st.drop(['steamspy_tags','negative_ratings'],axis=1))
print(st.columns)

print(st.corr())
sns.heatmap(st.corr())
sns.boxenplot()
sns.pairplot(st)


