#using computer_data dataset
#Import required libraries
#!pip install factor_analyzer
import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

comp= pd.read_csv("Computer_Data.csv")
comp.columns

#Dropping unnecessary columns
comp.drop(['cd','multi','premium'],axis=1, inplace=True)

#Dropping missing values rows
comp.dropna(inplace=True)
comp.info()
comp.head()

#Barlett's test
chi_square_value ,p_value=calculate_bartlett_sphericity(comp)
chi_square_value ,p_value

#Kaiser-Meyer-Oklin test
kmo_all, kmo_model=calculate_kmo(comp)
kmo_model

#Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.fit(comp)
#Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev

#Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.set_params(n_factors= 6, rotation="varimax")
fa.fit(comp)
loadings = fa.loadings_

#Get variance of each factors
fa.get_factor_variance()
