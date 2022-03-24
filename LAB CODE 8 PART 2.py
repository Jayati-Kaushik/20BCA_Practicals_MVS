# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 19:56:21 2022

@author: musai
"""

#Hierarchical Clustering, ,  

import os 

import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.preprocessing import normalize 

import scipy.cluster.hierarchy as shc 

from sklearn.cluster import AgglomerativeClustering 

 
 

 
 

os.chdir("C:/Users/prach/OneDrive/Desktop/Musai") 

data = pd.read_csv('Wholesale customers data.csv') 

print(data.head()) 

 
 

#normalize data so the scale of each varirable is same 

#if not done, model might become biased towards variables with higher magnitude (in this case fresh or milk) 

data_scaled = normalize(data) 

data_scaled = pd.DataFrame(data_scaled, columns=data.columns) 

print(data_scaled.head()) 

#similar scales 

 
 

#Dendogram to decide the number of clusters 

plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")  

d = shc.dendrogram(shc.linkage(data_scaled, method='ward')) 

#x=samples, y=distance between samples. threshold=6 

 
 

plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")  

d = shc.dendrogram(shc.linkage(data_scaled, method='ward')) 

plt.axhline(y=6, color='r', linestyle='--') 

#line divides forming 2 clusters  

 
 

#apply hierarchical clustering for 2 clusters 

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  

print(cluster.fit_predict(data_scaled)) 

#0=cluster 1, 1=cluster 2 

 
 

#visualize clusters 

plt.figure(figsize=(10, 7))  

plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_) 