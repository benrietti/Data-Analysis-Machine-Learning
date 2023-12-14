#!/usr/bin/env python
# coding: utf-8

# In[18]:


#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from apyori import apriori
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings('ignore')


# In[19]:


# Read CSV dynamically
file_path = 'teleco_market_basket.csv'
raw_data = pd.read_csv(teleco_market_basket.csv)


# In[20]:


#examine Data
df.head


# In[21]:


#look at datatypes
df.info()


# In[22]:


#shape dataset
df.shape


# In[23]:


#look for null values
df.isna().any()


# In[24]:


# Converting null to 0
newdf = df.fillna(0)


# In[25]:


#look for null values
newdf.isna().any()


# In[26]:


#clean data file
df.to_csv('df_prepared_TSK3.csv')


# In[27]:


# Converting dataframe into lists
transactions = []
for i in range(0, 15001):
    transactions.append([str(newdf.values[i,j]) for j in range(0, 20)])


# In[28]:


#apriori algorithm
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)


# In[29]:


# Turning results from apriori into a list
result = list(rules)
result


# In[30]:


#Turning list into a new dataframe
df = pd.DataFrame(columns=('Items','Antecedent','Consequent','Support','Confidence','Lift'))

Support =[]
Confidence = []
Lift = []
Items = []
Antecedent = []
Consequent=[]

for RelationRecord in result:
    for ordered_stat in RelationRecord.ordered_statistics:
        Support.append(RelationRecord.support)
        Items.append(RelationRecord.items)
        Antecedent.append(ordered_stat.items_base)
        Consequent.append(ordered_stat.items_add)
        Confidence.append(ordered_stat.confidence)
        Lift.append(ordered_stat.lift)

df['Items'] = list(map(set, Items))                                   
df['Antecedent'] = list(map(set, Antecedent))
df['Consequent'] = list(map(set, Consequent))
df['Support'] = Support
df['Confidence'] = Confidence
df['Lift']= Lift


# In[31]:


# Showing new dataframe
df


# In[32]:


# Showing the top rules by Support
df.sort_values(by ='Support', ascending = False, inplace = True)
df.head(10)


# In[33]:


# Showing the top rules by Confidence
df.sort_values(by ='Confidence', ascending = False, inplace = True)
df.head(10)


# In[34]:


# Showing the top rules by Lift
df.sort_values(by ='Lift', ascending = False, inplace = True)
df.head(10)


# In[ ]:





# In[ ]:




