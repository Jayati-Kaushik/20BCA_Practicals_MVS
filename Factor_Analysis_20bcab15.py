# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 09:12:32 2022

@author: joelp
"""

import os
import pandas as pd
from factor_analyzer import FactorAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
#from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

os.chdir("C:E:/JoelPaul/Desktop/Rstudios")
df=pd.read_csv('bfi.csv')
#dropping the non-numeric columns 
df.drop(['Gender','Education','Age'],axis=1,inplace=True) 

#drop missing values rows
df.dropna(inplace=True)
#df.fillna(0) #df.replace(np.nan,0)

