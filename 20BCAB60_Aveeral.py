

import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

os.chdir("C:/Users/Aveeral/Documents")
df=pd.read_csv('CarPrice.csv')
df.info()
df.drop(['car_ID','CarName'],axis=1,inplace=True)
df.info()
# Converting the categorical data into continous was done manually using FIND AND REPLACE in MS Excel.

# Checking the correlation

# Kaiser-Meyer-Olkin (KMO) Test
kmo_all,kmo_model=calculate_kmo(df)
print(kmo_model)


#Choosing the number of factors

fa = FactorAnalyzer()
fa.fit(df)
eigen_values, vectors = fa.get_eigenvalues()
print(vectors)


# Create scree plot using matplotlib
plt.scatter(range(1,df.shape[1]+1),vectors)
plt.plot(range(1,df.shape[1]+1),vectors)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.set_params(n_factors=6, rotation='varimax')
fa.fit(df)
loadings = fa.loadings_
print(loadings)


# Get variance of each factors
print(fa.get_factor_variance())

# Total 58% cumulative Variance is explained by the 3 factors.