# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:12:07 2022

@author: Joel Thomas
"""
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import pandas as pd
os.chdir("D:\Dump")
data=pd.read_csv('insurance.csv')
data.drop(['sex','smoker','region'],axis=1,inplace=True)
print(data.head())
print(data.dtypes)
print(data.shape)
print(data.head(15))
data.info()

#bartlett test
chi_square_value,p_value=calculate_bartlett_sphericity(data)
print(chi_square_value, p_value)

#kmo test
kmo_all,kmo_model=calculate_kmo(data)
print(kmo_model)

#eigen values
fa = FactorAnalyzer()
fa.fit(data)
eigen_values, vect = fa.get_eigenvalues()
print(vect)

#performing factor analysis
fa = FactorAnalyzer(4, rotation='varimax')
x=fa.fit(data)
loads = fa.loadings_
print(loads)

# Get variance of each factors
print(fa.get_factor_variance())
