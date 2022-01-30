#!/usr/bin/env python
# coding: utf-8

# In[2]:


from operator import imod
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


# In[3]:


data = make_blobs(n_samples = 200,n_features=2,centers=4,cluster_std=1.6,random_state=50)


# In[4]:


points = data[0]


# In[6]:


dendrogram = sch.dendrogram(sch.linkage(points,method="ward"))


# In[7]:


plt.scatter(data[0][:,0],data[0][:,1])


# In[10]:


hie= AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="ward")


# In[11]:


y_hie  = hie.fit_predict(points)


# In[13]:


plt.scatter(points[y_hie==0,0],points[y_hie==0,1],s = 100,c ='red')


# In[18]:


plt.scatter(points[y_hie==0,0],points[y_hie==0,1],s = 100,c ='red')
plt.scatter(points[y_hie==1,0],points[y_hie==1,1],s = 100,c ='green')
plt.scatter(points[y_hie==2,0],points[y_hie==2,1],s = 100,c ='yellow')
plt.scatter(points[y_hie==3,0],points[y_hie==3,1],s = 100,c ='blue')


# In[ ]:




