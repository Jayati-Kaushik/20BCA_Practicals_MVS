# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 22:21:11 2022

@author: admin
"""

#using airline dataset
#pip install factor_analyzer
import os
import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

os.chdir("C:/Users/admin/Desktop/Datasets")
data1=pd.read_csv('airlinedata.csv')
data1.drop(['Arrival Delay in Minutes'], axis=1, inplace=True)
data1.info()
data1.head()

#Subset of the data, the 14 columns containing the survey answers
x =data1[data1.columns[7:20]] 
fa = FactorAnalyzer()
fa.fit(x, 10)

#Bartlett’s test 
chi_square_value,p_value=calculate_bartlett_sphericity(x)
print(chi_square_value, p_value)

#Kaiser-Meyer-Olkin Test
kmo_all,kmo_model=calculate_kmo(x)
print(kmo_model)

#Get Eigen values and plot them
ev, v = fa.get_eigenvalues()
print(ev)
plt.scatter(range(1,x.shape[1]+1),ev)
plt.plot(range(1,x.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer(4, rotation='varimax')
fa.fit(x)
loads = fa.loadings_
print(loads)

# Get variance of each factors
print(fa.get_factor_variance())