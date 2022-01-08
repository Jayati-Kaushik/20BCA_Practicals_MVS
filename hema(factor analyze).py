# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 17:08:18 2022

@author: S Hema
"""

import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
df= pd.read_csv("bfi.csv")
df.column
     index(['A1', 'A2', 'A3', 'A4', 'A5', 'C1', 'C2', 'C3', 'C4', 'C5', 'E1', 'E2',
       'E3', 'E4', 'E5', 'N1', 'N2', 'N3', 'N4', 'N5', 'O1', 'O2', 'O3', 'O4',
       'O5', 'gender', 'education', 'age'],
      dtype='object')
df.drop(['gender', 'education', 'age'],axis=1,inplace=True)
df.dropna(inplace=True)
df.info()
df.head()
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df)
chi_square_value, p_value #pvalue is 0
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)
kmo_model #0.8486452309468382 is a good value
fa = FactorAnalyzer()
fa.analyze(df, 25, rotation=None)
ev, v = fa.get_eigenvalues()
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
fa = FactorAnalyzer()
fa.analyze(df, 6, rotation="varimax")
fa.loadings
fa = FactorAnalyzer()
fa.analyze(df, 5, rotation="varimax")
fa.loadings  