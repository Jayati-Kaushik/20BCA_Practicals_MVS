# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 08:54:58 2022

@author: Deepak SK
"""

from sklearn.datasets import load_digits
from sklearn.decomposition import FactorAnalysis
import os
import pandas as pd
import numpy as np

os.chdir("C:/Users/Deepak SK/Desktop/3rd sem/datasets")
d1 = pd.read_csv("covid_19_india.csv")
print(d1.head())
print(d1.dtypes)
print(d1.info())
X, _ = load_digits(return_X_y=True)
fa_Analysis = FactorAnalysis(n_components=8, random_state=123)
X_fa_Analysis  = fa_Analysis.fit_transform(X)
print(X_fa_Analysis.shape)