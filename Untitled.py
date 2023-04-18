#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd


# In[9]:


df=pd.read_excel("ECG data_07.01.2023.xlsx")


# In[10]:


df.head()


# In[12]:


df.shape


# In[13]:


df.describe()


# In[14]:


df.info()


# In[15]:


df.boxplot()


# In[17]:


df['patientAge'].unique() 


# In[18]:


df.median()


# In[19]:


df.kurtosis


# In[20]:


df.skew()


# In[21]:


df.corr()


# In[22]:


df.cov()


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns
correlation = df.corr()
plt.figure(figsize=(16, 8))
sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.show()


# In[30]:


df.hist()


# In[63]:


x = df[['P_wave','Q_wave','R_wave']]
y = df['PR_Interval']

import sklearn as sk
import sklearn.metrics as skl
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score


# In[64]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)
#random forest
nv =RandomForestRegressor(n_estimators = 1000, random_state = 42) # create a classifier
#naive_bayes.fit(X_train , y_train)
nv.fit(x_train,y_train)
x_test1=0.98
y_pred = nv.predict(x_test)
print(y_pred)


# In[59]:


print("Accuracy: ", r2_score(y_train, nv.predict(x_train)))


# In[67]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)
#decision tree
nv =svm.SVR() # create a classifier
#naive_bayes.fit(X_train , y_train)
nv.fit(x_train,y_train)

y_pred = nv.predict(x_test)
print(y_pred)


# In[73]:


print("Accuracy: ", r2_score(y_train, nv.predict(x_train)))


# In[74]:


from sklearn.neighbors import KNeighborsRegressor


# In[75]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)
#decision tree
nv =KNeighborsRegressor(n_neighbors=2) # create a classifier
#naive_bayes.fit(X_train , y_train)
nv.fit(x_train,y_train)

y_pred = nv.predict(x_test)
print(y_pred)


# In[76]:


print("Accuracy: ", r2_score(y_train, nv.predict(x_train)))


# In[ ]:




