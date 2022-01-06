from sklearn.datasets import load_digits
from sklearn.decomposition import FactorAnalysis
import os
import pandas as pd
import numpy as np

os.chdir("C:/Users/Seizal Pathania/Downloads/Rfiles")
d1 = pd.read_csv("covid data.csv")
print(d1.head())
print(d1.dtypes)
print(d1.info())
X, _ = load_digits(return_X_y=True)
fa_Analysis = FactorAnalysis(n_components=8, random_state=123)
X_fa_Analysis  = fa_Analysis.fit_transform(X)
print(X_fa_Analysis.shape)
