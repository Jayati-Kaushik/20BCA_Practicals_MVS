import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram, linkage 
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv('USArrests2.csv')
data.head()
data.info()
data.drop(['Unnamed: 0'],axis=1, inplace=True)
data.info()
data= np.array(data)

Z= linkage(data, method= "ward")
dendro= dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distance')
plt.show()

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  

labels = cluster.fit_predict(data)
plt.figure(figsize=(8, 5))
plt.scatter(data[labels == 0 , 0], data[labels == 0 , 1], c= 'red') 
plt.scatter(data[labels == 1 , 0], data[labels == 1 , 1], c= 'blue') 
plt.scatter(data[labels == 2 , 0], data[labels == 2 , 1], c= 'black') 
plt.scatter(data[labels == 3 , 0], data[labels == 3 , 1], c= 'green') 
plt.scatter(data[labels == 4 , 0], data[labels == 4 , 1], c= 'orange') 
plt.show()