# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:17:43 2022

@author: ACER
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

os.chdir("C:/Users/ACER/Downloads")
df=pd.read_csv('breast_cancer.csv')
df.info()
df.drop(['car_ID','CarName'],axis=1,inplace=True)
df.info()


