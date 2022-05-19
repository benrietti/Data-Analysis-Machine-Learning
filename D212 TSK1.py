#!/usr/bin/env python
# coding: utf-8

# In[80]:


#imports
import pandas as pd

import numpy as np

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

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

import kneed
from kneed import KneeLocator


import warnings
warnings.filterwarnings('ignore')


# In[81]:


#read csv
df= pd.read_csv('churn_clean.csv')


# In[82]:


#examine Data
df.head


# In[83]:


#look at datatypes
df.info()


# In[84]:


#drop columns not necessary for model
df=df.drop(columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State',
       'County', 'Zip', 'Lat', 'Lng', 'Population', 'Area', 'TimeZone', 'Job', 'Marital', 'Gender', 'Churn',
       'Techie', 'Contract', 'Port_modem', 'Tablet', 'InternetService',
       'Phone', 'Multiple', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'Item1', 'Item2', 'Item3', 'Item4', 'Item5',
       'Item6', 'Item7', 'Item8'])


# In[85]:


#evaluate updated dataset
df.info()


# In[86]:


#shape new dataset to see changes
df.shape


# In[87]:


#look for null values
df.isnull().sum()


# In[88]:


#look for duplicates
df.duplicated().sum()


# In[89]:


#summary stats
df.describe()


# In[90]:


#boxplots and scatterplots for outliers
sns.boxplot('Tenure', data=df)


# In[91]:


sns.boxplot('MonthlyCharge', data=df)


# In[92]:


sns.boxplot('Bandwidth_GB_Year', data=df)


# In[93]:


sns.boxplot('Outage_sec_perweek', data=df)
#although outliers are observed, the outliers does not seem extrenous that will affect the outcome of the algorithm


# In[94]:


sns.boxplot('Children', data=df)
#although outliers are observed, the outliers does not seem extrenous that will affect the outcome of the algorithm


# In[95]:


sns.boxplot('Income', data=df)
#although outliers are observed, the outliers does not seem extrenous that will affect the outcome of the algorithm


# In[96]:


sns.boxplot('Yearly_equip_failure', data=df)
#although outliers are observed, the outliers does not seem extrenous that will affect the outcome of the algorithm


# In[97]:


sns.boxplot('Email', data=df)
#although outliers are observed, the outliers does not seem extrenous that will affect the outcome of the algorithm


# In[98]:


sns.boxplot('Contacts', data=df)
#although outliers are observed, the outliers does not seem extrenous that will affect the outcome of the algorithm


# In[99]:


sns.boxplot('Age', data=df)


# In[100]:


sns.scatterplot(x=df['Outage_sec_perweek'],y=df['Tenure'])
plt.show()


# In[101]:


sns.scatterplot(x=df['MonthlyCharge'],y=df['Tenure'])
plt.show()


# In[102]:


sns.scatterplot(x=df['Bandwidth_GB_Year'],y=df['MonthlyCharge'])
plt.show()


# In[103]:


sns.scatterplot(x=df['MonthlyCharge'],y=df['Income'])
plt.show()


# In[104]:


sns.scatterplot(x=df['Income'],y=df['Tenure'])
plt.show()


# In[105]:


sns.scatterplot(x=df['Bandwidth_GB_Year'],y=df['Tenure'])
plt.show()


# In[106]:


#clean and prepared data file
df.to_csv('df_prepared_TSK1.csv')


# In[107]:


# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(df)


# In[108]:


# Create a PCA instance: pca
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X_std)


# In[109]:


# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


# In[110]:


# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)


# In[111]:


plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


# In[112]:


ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[113]:


X = df[["Tenure","MonthlyCharge"]]
#Visualise data points
plt.scatter(X["Tenure"],X["MonthlyCharge"],c='black')
plt.xlabel('Tenure')
plt.ylabel('MonthlyCharge (In Thousands)')
plt.show()


# In[114]:


#number of clusters
K=3

# Select random observation as centroids
Centroids = (X.sample(n=K))
plt.scatter(X["Tenure"],X["MonthlyCharge"],c='black')
plt.scatter(Centroids["Tenure"],Centroids["MonthlyCharge"],c='red')
plt.xlabel('Tenure')
plt.ylabel('MonthlyCharge (In Thousands)')
plt.show()


# In[115]:


# Step 3 - Assign all the points to the closest cluster centroid
# Step 4 - Recompute centroids of newly formed clusters
# Step 5 - Repeat step 3 and 4

diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Tenure"]-row_d["Tenure"])**2
            d2=(row_c["MonthlyCharge"]-row_d["MonthlyCharge"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["MonthlyCharge","Tenure"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['MonthlyCharge'] - Centroids['MonthlyCharge']).sum() + (Centroids_new['Tenure'] - Centroids['Tenure']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["MonthlyCharge","Tenure"]]


# In[116]:


color=['blue','green','cyan']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Tenure"],data["MonthlyCharge"],c=color[k])
plt.scatter(Centroids["Tenure"],Centroids["MonthlyCharge"],c='red')
plt.xlabel('Tenure')
plt.ylabel('MonthlyCharge (In Thousands)')
plt.show()


# In[ ]:




