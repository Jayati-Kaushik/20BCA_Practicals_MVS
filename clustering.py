# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 21:09:15 2022

@author: Harshitha
"""

import os
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.cluster
from scipy.cluster.hierarchy import dendrogram, linkage
os.chdir("C:/Users/Harshitha/Downloads")
df = pd.read_csv("creditcard_clustering")
print(df.head())
#df = df[:,:]
Z= linkage(df, method = "ward")
dendro= dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Eculidean distance')
plt.show()
ac = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="ward")
labels = ac.fit_predict(df)
plt.figure(figsize = (8,5))
plt.scatter(df[labels == 0 , 0 ], df[labels == 0 , 1], c = 'red')
plt.scatter(df[labels == 1 , 0 ], df[labels == 1 , 1], c = 'blue')
plt.scatter(df[labels == 2 , 0 ], df[labels == 2 , 1], c = 'green')
plt.scatter(df[labels == 3 , 0 ], df[labels == 3 , 1], c = 'black')
plt.scatter(df[labels == 4 , 0 ], df[labels == 4 , 1], c = 'orange')
plt.show()