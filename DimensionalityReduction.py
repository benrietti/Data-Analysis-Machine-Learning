#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports for dimensionality reduction methods-PCA
import pandas as pd
from pandas import Series, DataFrame

import numpy as np

import os 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import scipy.stats as scs
import prince
import pickle

plt.style.use('ggplot')

import sklearn 
from sklearn import preprocessing
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn import decomposition 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, explained_variance_score, confusion_matrix, accuracy_score, classification_report, log_loss
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn import discriminant_analysis 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plotly.express as px


# To change scientific numbers to float
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Increases the size of sns plots
sns.set(rc={'figure.figsize':(8,6)})

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

import kneed
from kneed import KneeLocator


import warnings
warnings.filterwarnings('ignore')


# In[2]:


#read csv
df= pd.read_csv('churn_clean.csv')


# In[3]:


df.head


# In[4]:


#look at datatypes
df.info()


# In[5]:


#shape dataset
df.shape


# In[6]:


#look for null values
df.isnull().sum()


# In[7]:


#look for duplicates
df.duplicated().sum()


# In[8]:


#summary stats
df.describe()


# In[9]:


df.columns


# In[10]:


#drop columns not necessary for model
xdf=df.drop(columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State',
       'County', 'Zip', 'Lat', 'Lng', 'Area', 'TimeZone', 'Job', 'Marital', 'Gender', 'Churn',
       'Techie', 'Contract', 'Port_modem', 'Tablet', 'InternetService',
       'Phone', 'Multiple', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Children', 'Email',
       'PaperlessBilling', 'PaymentMethod', 'Item1', 'Item2', 'Item3', 'Item4', 'Item5',
       'Item6', 'Item7', 'Item8'])


# In[11]:


xdf.dtypes


# In[12]:


#boxplots and scatterplots for outliers and brief analysis 
sns.boxplot('Population', data=df)


# In[13]:


sns.boxplot('MonthlyCharge', data=df)


# In[14]:


sns.boxplot('Bandwidth_GB_Year', data=df)


# In[15]:


sns.boxplot('Outage_sec_perweek', data=df)
#although outliers are observed, the outliers does not seem extrenous that will affect the outcome of the algorithm


# In[16]:


sns.boxplot('Income', data=df)
#although outliers are observed, the outliers does not seem extrenous that will affect the outcome of the algorithm


# In[17]:


sns.boxplot('Yearly_equip_failure', data=df)
#although outliers are observed, the outliers does not seem extrenous that will affect the outcome of the algorithm


# In[18]:


sns.boxplot('Contacts', data=df)
#although outliers are observed, the outliers does not seem extrenous that will affect the outcome of the algorithm


# In[19]:


sns.boxplot('Age', data=df)


# In[20]:


sns.scatterplot(x=df['Outage_sec_perweek'],y=df['Tenure'])
plt.show()


# In[21]:


sns.scatterplot(x=df['MonthlyCharge'],y=df['Tenure'])
plt.show()


# In[22]:


sns.scatterplot(x=df['Bandwidth_GB_Year'],y=df['MonthlyCharge'])
plt.show()


# In[23]:


sns.scatterplot(x=df['MonthlyCharge'],y=df['Income'])
plt.show()


# In[24]:


sns.scatterplot(x=df['Income'],y=df['Tenure'])
plt.show()


# In[25]:


sns.scatterplot(x=df['Bandwidth_GB_Year'],y=df['Tenure'])
plt.show()


# In[26]:


scaler = MinMaxScaler()
xdf[xdf.columns] = scaler.fit_transform(xdf[xdf.columns])


# In[27]:


xdf


# In[28]:


X_train = xdf.values


# In[29]:


# Trying with Dimentionality reduction and then Kmeans
n_components = X_train.shape[1]


# In[30]:


# Running PCA with all components
pca = PCA(n_components=n_components, random_state = 453)
X_r = pca.fit(X_train).transform(X_train)


# In[31]:


# Calculating the 95% Variance
total_variance = sum(pca.explained_variance_)
print("Total Variance in our dataset is: ", total_variance)
var_95 = total_variance * 0.95
print("The 95% variance we want to have is: ", var_95)
print("")


# In[32]:


#Creating a df with the components and explained variance
a = zip(range(0,n_components), pca.explained_variance_)
a = pd.DataFrame(a, columns=["PCA Comp", "Explained Variance"])


# In[33]:


#Trying to hit 95%
print("Variance explain with 2 n_compononets: ", sum(a["Explained Variance"][0:2]))
print("Variance explain with 5 n_compononets: ", sum(a["Explained Variance"][0:5]))
print("Variance explain with 6 n_compononets: ", sum(a["Explained Variance"][0:6]))
print("Variance explain with 7 n_compononets: ", sum(a["Explained Variance"][0:7]))
print("Variance explain with 8 n_compononets: ", sum(a["Explained Variance"][0:8]))
print("Variance explain with 9 n_compononets: ", sum(a["Explained Variance"][0:9]))
print("Variance explain with 10 n_compononets: ", sum(a["Explained Variance"][0:10]))

print("Variance explain with 20 n_compononets: ", sum(a["Explained Variance"][0:20]))
print("Variance explain with 25 n_compononets: ", sum(a["Explained Variance"][0:25]))
print("Variance explain with 26 n_compononets: ", sum(a["Explained Variance"][0:26]))
print("Variance explain with 27 n_compononets: ", sum(a["Explained Variance"][0:27]))
print("Variance explain with 30 n_compononets: ", sum(a["Explained Variance"][0:30]))
print("Variance explain with 35 n_compononets: ", sum(a["Explained Variance"][0:35]))
print("Variance explain with 40 n_compononets: ", sum(a["Explained Variance"][0:40]))
print("Variance explain with 41 n_compononets: ", sum(a["Explained Variance"][0:41]))
print("Variance explain with 42 n_compononets: ", sum(a["Explained Variance"][0:42]))
print("Variance explain with 44 n_compononets: ", sum(a["Explained Variance"][0:44]))
print("Variance explain with 45 n_compononets: ", sum(a["Explained Variance"][0:45]))
print("Variance explain with 50 n_compononets: ", sum(a["Explained Variance"][0:50]))
print("Variance explain with 53 n_compononets: ", sum(a["Explained Variance"][0:53]))
print("Variance explain with 55 n_compononets: ", sum(a["Explained Variance"][0:55]))
print("Variance explain with 60 n_compononets: ", sum(a["Explained Variance"][0:60]))
print("Variance explain with 70 n_compononets: ", sum(a["Explained Variance"][0:70]))
print("Variance explain with 80 n_compononets: ", sum(a["Explained Variance"][0:80]))


# In[34]:


#Plotting the Data
plt.figure(1, figsize=(14, 8))
plt.plot(pca.explained_variance_ratio_, linewidth=2, c="r")
plt.xlabel('n_components')
plt.ylabel('explained_ratio_')


# In[35]:


#Plotting line with 95% e.v.
plt.axvline(7,linestyle=':', label='n_components - 95% explained', c ="blue")
plt.legend(prop=dict(size=12))


# In[36]:


#adding arrow
plt.annotate('7 eigenvectors used to explain 95% variance', xy=(7, pca.explained_variance_ratio_[7]), 
             xytext=(7, pca.explained_variance_ratio_[7]),
            arrowprops=dict(facecolor='blue', shrink=0.05))


# In[37]:


plt.show()


# In[38]:


pca = PCA(n_components = 7)


# In[39]:


pca.fit(X_train)


# In[40]:


x_pca = pca.transform(X_train)

print(x_pca.shape)

print(X_train.shape)


# In[41]:


plt.scatter(x_pca[:,0],x_pca[:,1])

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component')


# In[42]:


pca.components_ 


# In[43]:


variance = pca.explained_variance_ratio_


# In[44]:


variance


# In[45]:


x_pca_df=pd.DataFrame(x_pca,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7'])
x_pca_df


# In[46]:


variance = np.insert(variance, 0,0)


# In[47]:


cumulative_variance = np.cumsum(np.round(variance, decimals=3))


# In[48]:


pc_df = pd.DataFrame(['','PC1', 'PC2', 'PC3','PC4','PC5','PC6','PC7'], columns=['PC'])
explained_variance_df = pd.DataFrame(variance, columns=['Explained Variance'])
cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=['Cumulative Variance'])


# In[49]:


df_explained_variance = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis=1)
df_explained_variance


# In[50]:


# https://plotly.com/python/bar-charts/
# Scree plot that shows the variance of the PC components.
fig = px.bar(df_explained_variance, 
             x='PC', y='Explained Variance',
             text='Explained Variance',
             width=800)


# In[51]:


fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig.show()


# In[52]:


# https://plotly.com/python/creating-and-updating-figures/


# In[53]:


import plotly.graph_objects as go


# In[54]:


fig = go.Figure()


# In[55]:


fig.add_trace(
    go.Scatter(
        x=df_explained_variance['PC'],
        y=df_explained_variance['Cumulative Variance'],
        marker=dict(size=15, color="LightSeaGreen")
    ))


# In[56]:


fig.add_trace(
    go.Bar(
        x=df_explained_variance['PC'],
        y=df_explained_variance['Explained Variance'],
        marker=dict(color="RoyalBlue")
    ))


# In[57]:


fig.show()


# In[58]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[59]:


fig = make_subplots(rows=2, cols=1)


# In[60]:


fig.add_trace(
    go.Scatter(
        x=df_explained_variance['PC'],
        y=df_explained_variance['Cumulative Variance'],
        marker=dict(size=15, color="LightSeaGreen")
    ), row=1, col=1
    )


# In[61]:


fig.add_trace(
    go.Bar(
        x=df_explained_variance['PC'],
        y=df_explained_variance['Explained Variance'],
        marker=dict(color="RoyalBlue"),
    ), row=2, col=1
    )


# In[62]:


fig = px.scatter_3d(x_pca_df, x='PC1', y='PC2', z='PC3',
              color='PC1')


# In[63]:


fig.show()


# In[64]:


xdf_pca = pd.concat([df, x_pca_df], axis=1)


# In[65]:


xdf_pca


# In[66]:


fig = px.scatter_3d(xdf_pca, x='PC1', y='PC2', z='PC3',
              color='Churn')


# In[67]:


fig.show()


# In[68]:


fig = px.scatter_3d(xdf_pca, x='PC1', y='PC2', z='PC3',
              color='Churn',
              symbol='Churn',
              opacity=0.5)


# In[69]:


# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))


# In[70]:


# https://plotly.com/python/templates/
#fig.update_layout(template='plotly_white') # "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"


# In[71]:


covMatrix = np.cov(xdf,bias=True)
print (covMatrix)


# In[73]:


#clean and prepared data file
xdf.to_csv('df_prepared_TSK2.csv')


# In[74]:


#clean and prepared data file
xdf_pca.to_csv('df_prepared_TSK2_PCA.csv')


# In[ ]:




