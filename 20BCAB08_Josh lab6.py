# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:08:28 2022

@author: Josh
"""

import pandas as pd
#pip install factor_analyzer
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
os.chdir("D:/NOTES/Datasets")
mydata = pd.read_csv('Computer_Data edited.csv')
print(mydata.head())
mydata.columns
mydata.dropna(inplace=True)
mydata.info()

#bartlett test
chi_square_value,p_value=calculate_bartlett_sphericity(mydata)
print(chi_square_value, p_value)
#kmo test
kmo_all,kmo_model=calculate_kmo(mydata)
print(kmo_model)


#eigen values
fa = FactorAnalyzer()
fa.fit(mydata)
eigen_values, vect = fa.get_eigenvalues()
print(vect)

# Create scree plot using matplotlib
plt.scatter(range(1,mydata.shape[1]+1),vect)
plt.plot(range(1,mydata.shape[1]+1),vect)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

#performing factor analysis
fa = FactorAnalyzer(4, rotation='varimax')
x=fa.fit(mydata)
loads = fa.loadings_
print(loads)

# Get variance of each factors
print(fa.get_factor_variance()) 
