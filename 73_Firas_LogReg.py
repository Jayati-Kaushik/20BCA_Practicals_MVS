#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("C:/Users/WinDows/OneDrive/Desktop/framingham.csv")
df


# In[4]:


df.nunique()


# In[5]:


df.isnull().sum()


# In[6]:


df.shape


# In[11]:


df1= df.dropna(how = 'any')
df1


# In[12]:


df1.isnull().sum()


# In[13]:


plt.scatter(df1.age,df1.diabetes,marker='+',color='red')


# In[14]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test =train_test_split(df1[['age']],df1.diabetes,train_size=0.7)


# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


model = LogisticRegression()


# In[20]:


model.fit(x_train,y_train)


# In[21]:


model.predict(x_test)


# In[22]:


model.predict_proba(x_test)


# In[ ]:




