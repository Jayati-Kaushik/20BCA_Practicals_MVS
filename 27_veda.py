# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:19:07 2022

@author: Dell
"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer


import matplotlib.pyplot as plt
import os

os.chdir("C:/Users/Dell/Downloads")
df= pd.read_csv('bikes_price.csv')
df.info()
df.drop(['age'],axis=1,inplace=True)
df.info()


#to calculate bartlett sphericity
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df)
chi_square_value, p_value


#to calculate Kaiser-Meyer-Olkin (KMO) Test
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)
kmo_model

# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.analyze(df, 25, rotation=None)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev


# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.analyze(df, 6, rotation="varimax")
fa.loadings

# Create factor analysis object and perform factor analysis using 5 factors
fa = FactorAnalyzer()
fa.analyze(df, 5, rotation="varimax")
fa.loadings

# Get variance of each factors
fa.get_factor_variance()