# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:30:25 2022

@author: Raisa
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn import preprocessing
mydata=pd.read_csv("C:/Users/Raisa/Documents/jerry(DS)/datasets/Walmart_Store_sales.csv")
print(mydata.shape)
print(mydata.head(10))
mydata.info()
print(mydata.describe())
print(mydata.dtypes)
data_normal = preprocessing.scale(mydata)
fa = FactorAnalysis(n_components = 1)
fa.fit(data_normal)
for score in fa.score_samples(data_normal):
    print(score) 
