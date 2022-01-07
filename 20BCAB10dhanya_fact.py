# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn import preprocessing
from sklearn.datasets import load_digits
os.chdir("C:/Users/DHANYA/Desktop/datascience dhanya")
data=pd.read_csv('titanic.csv')
data.drop(['Age'],axis=1,inplace=True)
print(data.head())
print(data.dtypes)
print(data.shape)
print(data.head(15))
data.info()
X, _ = load_digits(return_X_y=True)
fa_Analysis = FactorAnalysis(n_components=5, random_state=92)
X_fa_Analysis  = fa_Analysis.fit_transform(X)
print(X_fa_Analysis.shape)



