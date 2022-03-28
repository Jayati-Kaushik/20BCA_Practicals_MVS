# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:55:23 2021

@author: dell
"""

import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


# In[17]:


df = pd.read_csv("C:/Users/WinDows/OneDrive/Desktop/housing.csv")
df


# In[18]:


df.isnull().sum()


# In[19]:


df.head()


# In[20]:


df = df.drop(['date'],axis=1)


# In[21]:


df.head()


# In[22]:


reg = linear_model.LinearRegression()


# In[25]:


reg.fit(df[['age','latitude','stores']],df.price)


# In[27]:


reg.coef_


# In[28]:


reg.intercept_


# In[30]:


reg.predict([[16,24.98746,5]])


# In[32]:


reg.predict([[5,24.98746,12]])


# In[33]:


import seaborn as sns


# In[38]:


cor = df.corr
cor


# In[41]:





# In[ ]: