#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd

pd.options.display.max_rows = 9999

df = pd.read_csv('medical_raw_data.csv')


# In[7]:


df.info()


# In[8]:


print(df)


# In[9]:


df.duplicated()


# In[10]:


df.duplicated().sum()


# In[11]:


df.isnull().sum()


# In[12]:


df.hist()


# In[13]:


df['Children'].fillna(df['Children'].median(), inplace=True)


# In[14]:


df['Age'].fillna(df['Age'].mean(),inplace=True)


# In[15]:


df['Income'].fillna(df['Income'].median(),inplace=True)


# In[16]:


df['Soft_drink']= df ['Soft_drink'].fillna(df['Soft_drink'].mode()[0])


# In[17]:


df['Overweight'].fillna(df['Overweight'].mean(),inplace=True)


# In[18]:


df['Anxiety'].fillna(df['Anxiety'].mean(),inplace=True)


# In[19]:


df['Initial_days'].fillna(df['Initial_days'].mean(),inplace=True)


# In[20]:


df.isnull().sum()


# In[21]:


import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', '')


# In[22]:


data=df.select_dtypes(float, int)


# In[23]:


data.head(2)


# In[24]:


df['zscore_Age'] = (df.Age-df.Age.mean())/df.Age.std()
df['zscore_Income'] = (df.Income-df.Income.mean())/df.Income.std()
df['zscore_VitD_levels'] = (df.VitD_levels-df.VitD_levels.mean())/df.VitD_levels.std()
df['zscore_Doc_visits'] = (df.Doc_visits-df.Doc_visits.mean())/df.Doc_visits.std()
df['zscore_Full_meals_eaten'] = (df.Full_meals_eaten-df.Full_meals_eaten.mean())/df.Full_meals_eaten.std()
df['zscore_Initial_days'] = (df.Initial_days-df.Initial_days.mean())/df.Initial_days.std()
df['zscore_TotalCharge'] = (df.TotalCharge-df.TotalCharge.mean())/df.TotalCharge.std()
df['zscore_Additional_charges'] = (df.Additional_charges-df.Additional_charges.mean())/df.Additional_charges.std()
df.sample(2)


# In[25]:


outliers_z = df[(df.zscore_Age < -3) | (df.zscore_Age>3)]
outliers_z.shape
outliers_z = df[(df.zscore_Income < -3) | (df.zscore_Income>3)]
outliers_z.shape
outliers_z = df[(df.zscore_VitD_levels < -3) | (df.zscore_VitD_levels>3)]
outliers_z.shape
outliers_z = df[(df.zscore_Doc_visits < -3) | (df.zscore_Doc_visits >3)]
outliers_z.shape
outliers_z = df[(df.zscore_Full_meals_eaten < -3) | (df.zscore_Full_meals_eaten>3)]
outliers_z.shape
outliers_z = df[(df.zscore_Initial_days < -3) | (df.zscore_Initial_days>3)]
outliers_z.shape
outliers_z = df[(df.zscore_Additional_charges < -3) | (df.zscore_Additional_charges>3)]
outliers_z.shape
outliers_z = df[(df.zscore_TotalCharge < -3) | (df.zscore_TotalCharge>3)]
outliers_z.shape


# In[26]:


from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# In[27]:


pca = df [[ 'Age', 'Children', 'Income', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten', 'Initial_days', 'TotalCharge', 'Additional_charges', 'Anxiety', 'Overweight']] 


# In[28]:


pca_normalized = (pca-pca.mean())/pca.std()


# In[29]:


pca = PCA (n_components=pca.shape[1])


# In[30]:


pca.fit(pca_normalized)


# In[31]:


pca2 = pd.DataFrame(pca.transform(pca_normalized),columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10', 'pc11'])


# In[32]:


loadings = pd.DataFrame(pca.components_.T,
columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10', 'pc11'],
index=pca_normalized.columns)
loadings


# In[33]:


cov_matrix = np.dot(pca_normalized.T,pca_normalized) / pca_normalized.shape[0]
eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for eigenvector in pca.components_]


# In[34]:


plt.plot(eigenvalues)
plt.xlabel('number of components')
plt.ylabel('eigenvalues')
plt.show()


# In[35]:


df.to_csv('F_ProjectD206_Resubmission.csv')


# In[ ]:





# In[ ]:




