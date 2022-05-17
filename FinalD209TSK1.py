#!/usr/bin/env python
# coding: utf-8

# In[552]:


#import libraries and packages
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# In[553]:


#upload data set
df= pd.read_csv('churn_clean.csv')


# In[554]:


#view and shape dataset
df


# In[555]:


#rename survey columns
df.rename(columns={'Item1':'TimelyResponse','Item2':'Fixes','Item3':'Replacements','Item4':'Reliability','Item5':'Options','Item6':'Respecfulness','Item7':'Courteous','Item8':'Listening'}, inplace=True)


# In[556]:


#print column names
df.columns


# In[557]:


#summary statistics
df. describe()


# In[558]:


#drop unnecessary columns
df=df.drop(columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 
                            'State', 'County', 'Zip', 'Lat', 'Lng', 'Population', 
                            'Area', 'TimeZone', 'Job', 'Marital', 'PaymentMethod'])


# In[559]:


#null values
df.isnull().sum()


# In[560]:


#print data types
df.dtypes


# In[561]:


#dummy variables for categorical data 
df = pd.get_dummies(df)


# In[562]:


#print data set
df


# In[563]:


#summary stats
df.describe()


# In[564]:


# Column names
df.columns


# In[565]:


#drop unnecessary columns
df=df.drop(columns=['Gender_Female','Gender_Nonbinary', 'Churn_No', 'Techie_No','Contract_Month-to-month', 'Contract_One year','Port_modem_No', 'Tablet_No','InternetService_DSL','InternetService_None','Phone_No','Multiple_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No','PaperlessBilling_No'])
df.describe()


# In[566]:


#review columns
df.columns


# In[567]:


#review columns
df.dtypes


# In[568]:


#histograms and boxplots
df[['Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts',
       'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']].hist()
plt.show()


# In[569]:


df[[ 'TimelyResponse', 'Fixes', 'Replacements', 'Reliability', 'Options',
       'Respecfulness', 'Courteous', 'Listening', 'Gender_Male',
       'Techie_Yes', 'Contract_Two Year', 'Port_modem_Yes']].hist()
plt.show()


# In[570]:


df[['Tablet_Yes',
       'InternetService_Fiber Optic', 'Phone_Yes', 'Multiple_Yes',
       'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
       'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
       'PaperlessBilling_Yes', 'Churn_Yes']].hist()
plt.show()


# In[571]:


sns.boxplot('Tenure', data=df)


# In[572]:


sns.boxplot('MonthlyCharge', data=df)


# In[573]:


sns.boxplot('Bandwidth_GB_Year', data=df)


# In[574]:


sns.boxplot('Outage_sec_perweek', data=df)


# In[575]:


sns.boxplot('Children', data=df)


# In[576]:


sns.boxplot('Income', data=df)


# In[577]:


sns.boxplot('Yearly_equip_failure', data=df)


# In[578]:


sns.boxplot('Email', data=df)


# In[579]:


sns.boxplot('Contacts', data=df)


# In[580]:


sns.boxplot('Age', data=df)


# In[581]:


sns.scatterplot(x=df['Outage_sec_perweek'],y=df['Churn_Yes'])
plt.show()


# In[582]:


sns.scatterplot(x=df['Bandwidth_GB_Year'],y=df['Churn_Yes'])
plt.show()


# In[583]:


sns.scatterplot(x=df['MonthlyCharge'],y=df['Churn_Yes'])
plt.show()


# In[584]:


sns.scatterplot(x=df['Income'],y=df['Churn_Yes'])
plt.show()


# In[585]:


sns.scatterplot(x=df['Email'],y=df['Churn_Yes'])
plt.show()


# In[586]:


sns.scatterplot(x=df['Yearly_equip_failure'],y=df['Churn_Yes'])
plt.show()


# In[587]:


sns.scatterplot(x=df['Contacts'],y=df['Churn_Yes'])
plt.show()


# In[588]:


sns.scatterplot(x=df['Tenure'],y=df['Churn_Yes'])
plt.show()


# In[589]:


#create new df
df = df[['Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts',
       'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year',
       'TimelyResponse', 'Fixes', 'Replacements', 'Reliability', 'Options',
       'Respecfulness', 'Courteous', 'Listening', 'Gender_Male',
       'Techie_Yes', 'Contract_Two Year', 'Port_modem_Yes', 'Tablet_Yes',
       'InternetService_Fiber Optic', 'Phone_Yes', 'Multiple_Yes',
       'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
       'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
       'PaperlessBilling_Yes', 'Churn_Yes']]


# In[590]:


#export to csv
df.to_csv('df_prepared.csv')


# In[591]:


#split 
X = df.drop('Churn_Yes', axis=1).values
y = df['Churn_Yes'].values


# In[592]:


#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)


# In[593]:


#split
y_pred = knn.predict(X_test)
knn.score(X_test, y_test)


# In[594]:


from sklearn.metrics import accuracy_score


# In[595]:


#accuracy score
print(accuracy_score(y_test, y_pred))


# In[596]:


#knn 
knn.score(X_test, y_test)


# In[597]:


#knn
knn.score(X_train, y_train)


# In[598]:


#Confusion matrix/classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[599]:


#accuracy score
accuracy_score(y_pred, y_test)


# In[600]:


#scaling the dataset to see if a better accuracy score is achieved
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X, y, test_size=0.2, random_state=42)
knn_scaled = pipeline.fit(X_train_scaled, y_train_scaled)
y_pred = pipeline.predict(X_test_scaled)
print('Accuracy score after scaling: {}'.format(accuracy_score(y_test_scaled, y_pred)))


# In[601]:


# confusion matrx and report after scaling
print(confusion_matrix(y_test_scaled, y_pred))
print(classification_report(y_test_scaled, y_pred))


# In[602]:


from sklearn.model_selection import GridSearchCV


# In[603]:


param_grid = {'n_neighbors': np.arange(1,50)}


# In[604]:


knn = KNeighborsClassifier(n_neighbors=7)


# In[605]:


knn_cv = GridSearchCV(knn, param_grid, cv=5)


# In[606]:


#knn
knn_cv.fit(X_train,y_train)
knn_cv.best_params_
knn_cv.best_score_


# In[607]:


#AUC
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
print('AUC score: {}'.format(roc_auc_score(y_test, y_pred_prob)))


# In[608]:


#AUC cross-val
cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
print("AUC cross-val score: {}".format(cv_scores))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




