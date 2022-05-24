#!/usr/bin/env python
# coding: utf-8

# In[439]:


# Import Libraries and CSV


# In[440]:


import pandas as pd
import seaborn as sns
sns.set_theme(style="ticks", color_codes=True)
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('medical_clean.csv')


# In[472]:


df.shape


# In[473]:


# View date file


# In[474]:


df


# In[475]:


df.dtypes


# In[445]:


# Check for duplicate values


# In[446]:


df.duplicated().sum()


# In[447]:


# Check for null values


# In[448]:


df.isnull().sum()


# In[449]:


# Install and Import plotnine


# In[450]:


conda install -c conda-forge plotnine


# In[451]:


import plotnine as p9
from plotnine import ggplot, aes


# In[452]:


# Create two arrays with ratios


# In[453]:


Age_ratio = df[df.Age == "Age"].Age
Doc_visits_ratio = df[df.Doc_visits == "Doc_visits"].Doc_visits


# In[454]:


# Perform the two-sample t-test and print results


# In[455]:


t_result= stats.ttest_ind(Age_ratio, Doc_visits_ratio)
print(t_result)


# In[456]:


# Test significance


# In[457]:


alpha= 0.05
if (t_result[1] < alpha):
    print("Age and Doc_visits have different mean ratios")
else: print("No significant difference found")


# In[458]:


# Univariate Continous 
# Print density plot, mean, median, and mode of Full_meals_eaten


# In[459]:


print(p9.ggplot(df)+ p9.aes(x='Full_meals_eaten')+ p9.geom_density())
print(df.Full_meals_eaten.mean())
print(df.Full_meals_eaten.median())
print(df.Full_meals_eaten.mode())


# In[460]:


# Print density plot, mean, median, and mode of Income


# In[461]:


print(p9.ggplot(df)+ p9.aes(x='Income')+ p9.geom_density())
print(df.Income.mean())
print(df.Income.median())
print(df.Income.mode())


# In[462]:


# univariate categorical
# frequency tables


# In[463]:


pd.crosstab(index=df['Gender'], columns='count')


# In[464]:


pd.crosstab(index=df['Marital'], columns='count')


# In[465]:


# Convert categorical to number 


# In[466]:


df['Gender'].replace(['Male', 'Female', 'Nonbinary'],
                        [0, 1, 2], inplace=True)


# In[467]:


df['Marital'].replace(['Divorced', 'Married', 'Widowed', 'Seperated', 'Never Married'],
                        [0, 1, 2, 3, 4], inplace=True)


# In[468]:


# Bivariate 


# In[469]:


sns.barplot(x='Gender',y='Marital',data=df) 


# In[470]:


#boxplot


# In[480]:


sns.boxplot (x = "Full_meals_eaten", y = "VitD_levels", data=df)
plt.show()


# In[481]:


df.to_csv('D207_PA.csv')


# In[ ]:




