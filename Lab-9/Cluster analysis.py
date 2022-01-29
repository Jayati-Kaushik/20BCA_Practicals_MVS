import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#For creating a directory
import os
#Returning the current working directory
cwd = os.getcwd()
#Changing the current working directory
os.chdir("C:/Users/SONY/Desktop/Wholesale customers data")
customer=pd.read_csv('customers data.csv')
print(customer.head())
from sklearn.preprocessing import normalize
data_scaled = normalize(customer)
data_scaled = pd.DataFrame(data_scaled, columns=customer.columns)
data_scaled.head()
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
#Deciding a threshold of 6 and cutting the dendrogram
plt.axhline(y=6, color='r', linestyle='--')
#Applying hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)
#Visualizing the two clusters
plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_) 
