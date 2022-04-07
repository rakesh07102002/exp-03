#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd


# In[21]:


import numpy as np


# In[22]:


import seaborn as sns


# In[23]:


df=pd.read_csv("C:\\Users\\USER\\Downloads\\titanic_dataset.csv")


# In[24]:


df.info()


# In[25]:


df.head()


# In[26]:


df.isnull().sum()


# In[27]:


df.drop(columns=['Cabin'],inplace=True)


# In[28]:


df.info()


# In[29]:


df["Age"]=df["Age"].fillna(df["Age"].median())


# In[30]:


df.boxplot()


# In[31]:


df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])


# In[33]:


df["Embarked"].value_counts()


# In[34]:


df["Pclass"].value_counts()


# In[35]:


df["Survived"].value_counts()


# In[36]:


sns.countplot(x="Survived",data=df)


# In[37]:


sns.countplot(x="Sex",data=df)


# In[38]:


df.info()


# In[39]:


sns.displot(df["Fare"])


# In[40]:


sns.countplot(x="Pclass",hue="Survived",data=df)


# In[41]:


sns.countplot(x="Sex",hue="Survived",data=df)


# In[42]:


sns.displot(df[df["Survived"]==0]["Age"])


# In[43]:


pd.crosstab(df["Pclass"],df["Survived"])


# In[44]:


df.corr()


# In[45]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:




