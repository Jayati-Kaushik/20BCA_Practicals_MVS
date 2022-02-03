# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:08:02 2022

@author: Josh
"""

from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage 
import os

os.chdir("D:/NOTES/Datasets")
car_pr = pd.read_csv('CarPrice_Assignment.csv')
car_pr.drop(['car_ID','CarName'],axis=1,inplace=True)
car_pr.info()
xy = car_pr.iloc[:, [8,9]].values

Z = linkage(xy, method = "ward")
dendro = dendrogram(Z)
plt.title('Dendogram')
plt.ylabel('Euclidean distance')
plt.show()
ac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")

labels = ac.fit_predict(xy)
plt.figure(figsize = (8,5))
plt.scatter(xy[labels == 0,0] , xy[labels == 0,1], c= 'black')
plt.scatter(xy[labels == 1,0] , xy[labels == 1,1], c= 'blue')
plt.scatter(xy[labels == 2,0] , xy[labels == 2,1], c= 'green')
plt.scatter(xy[labels == 3,0] , xy[labels == 3,1], c= 'red')
plt.scatter(xy[labels == 4,0] , xy[labels == 4,1], c= 'orange')
plt.show()
