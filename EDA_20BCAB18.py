# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 21:09:10 2022

@author: Shahid
"""

import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

os.chdir("C:/Users/Shahid Khan/Downloads/steam.csv")
data = pd.read_csv('steam.csv')
data.head()
print(data.head())
print(data.tail())
data.describe()
print(data.describe())

#Checking for null values
print(data.isnull().sum())


print(data.head())

print(data.corr())
sns.heatmap(data.corr())
sns.pairplot(data)
