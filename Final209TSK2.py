#!/usr/bin/env python
# coding: utf-8

# In[170]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn
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


# In[171]:


#upload data set
df= pd.read_csv('churn_clean.csv')
df


# In[172]:


#rename survey columns
df.rename(columns={'Item1':'TimelyResponse','Item2':'Fixes','Item3':'Replacements','Item4':'Reliability','Item5':'Options','Item6':'Respecfulness','Item7':'Courteous','Item8':'Listening'}, inplace=True)


# In[173]:


#print column names
df.columns


# In[174]:


#summary statistics
df. describe()


# In[175]:


#drop columns
df=df.drop(columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 
                            'State', 'County', 'Zip', 'Lat', 'Lng', 'Population', 
                            'Area', 'TimeZone', 'Job', 'Marital', 'PaymentMethod'])


# In[176]:


#null values
df.isnull().sum()


# In[177]:


#print data types
df.dtypes


# In[178]:


#dummy variables
df = pd.get_dummies(df)


# In[179]:


#print data set
df


# In[180]:


#summary stats
df.describe()


# In[181]:


# Column names
df.columns


# In[182]:


#drop unnecessary columns
df=df.drop(columns=['Gender_Female','Gender_Nonbinary', 'Churn_No', 'Techie_No','Contract_Month-to-month', 'Contract_One year','Port_modem_No', 'Tablet_No','InternetService_DSL','InternetService_None','Phone_No','Multiple_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No','PaperlessBilling_No'])
df.describe()


# In[183]:


#review columns
df.columns


# In[184]:


#review columns
df.dtypes


# In[185]:


#histograms and boxplots
df[['Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts',
       'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']].hist()
plt.show()


# In[186]:


df[[ 'TimelyResponse', 'Fixes', 'Replacements', 'Reliability', 'Options',
       'Respecfulness', 'Courteous', 'Listening', 'Gender_Male',
       'Techie_Yes', 'Contract_Two Year', 'Port_modem_Yes']].hist()
plt.show()


# In[187]:


df[['Tablet_Yes',
       'InternetService_Fiber Optic', 'Phone_Yes', 'Multiple_Yes',
       'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
       'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
       'PaperlessBilling_Yes', 'Churn_Yes']].hist()
plt.show()


# In[188]:


sns.boxplot('Tenure', data=df)


# In[189]:


sns.boxplot('MonthlyCharge', data=df)


# In[190]:


sns.boxplot('Bandwidth_GB_Year', data=df)


# In[191]:


sns.boxplot('Outage_sec_perweek', data=df)


# In[192]:


sns.boxplot('Children', data=df)


# In[193]:


sns.boxplot('Income', data=df)


# In[194]:


sns.boxplot('Yearly_equip_failure', data=df)


# In[195]:


sns.boxplot('Email', data=df)


# In[196]:


sns.boxplot('Contacts', data=df)


# In[197]:


sns.boxplot('Age', data=df)


# In[198]:


sns.scatterplot(x=df['Outage_sec_perweek'],y=df['Churn_Yes'])
plt.show()


# In[199]:


sns.scatterplot(x=df['Bandwidth_GB_Year'],y=df['Churn_Yes'])
plt.show()


# In[200]:


sns.scatterplot(x=df['MonthlyCharge'],y=df['Churn_Yes'])
plt.show()


# In[201]:


sns.scatterplot(x=df['Income'],y=df['Churn_Yes'])
plt.show()


# In[202]:


sns.scatterplot(x=df['Email'],y=df['Churn_Yes'])
plt.show()


# In[203]:


sns.scatterplot(x=df['Yearly_equip_failure'],y=df['Churn_Yes'])
plt.show()


# In[204]:


sns.scatterplot(x=df['Contacts'],y=df['Churn_Yes'])
plt.show()


# In[205]:


sns.scatterplot(x=df['Tenure'],y=df['Churn_Yes'])
plt.show()


# In[206]:


# Create new df
df = df[['Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts',
       'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year',
       'TimelyResponse', 'Fixes', 'Replacements', 'Reliability', 'Options',
       'Respecfulness', 'Courteous', 'Listening', 'Gender_Male',
       'Techie_Yes', 'Contract_Two Year', 'Port_modem_Yes', 'Tablet_Yes',
       'InternetService_Fiber Optic', 'Phone_Yes', 'Multiple_Yes',
       'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
       'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
       'PaperlessBilling_Yes', 'Churn_Yes']]


# In[207]:


#export to csv
df.to_csv('df_preparedD209TSK2.csv')


# In[208]:


X = df.drop('Churn_Yes', axis=1).values
y = df['Churn_Yes'].values


# In[209]:


SEED=1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


# In[210]:


print('X train shape:', X_train.shape)
print('X test shape:', X_test.shape)
print('y train shape:', y_train.shape)
print('y test shape:', y_test.shape)


# In[211]:


rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=SEED)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[212]:


#RMSE of model
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))


# In[213]:


#feature importance
import pandas as pd
import matplotlib.pyplot as plt
importances_rf = pd.Series(rf.feature_importances_)
sorted_importances_rf = importances_rf.sort_values()
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()


# In[214]:


for i, item in enumerate(rf.feature_importances_):
    print('{0:s}: {1:.2f}'.format(df.columns[i], item))


# In[215]:


rf.get_params()


# In[216]:


#accuracy of train and test data with MAE
rf.fit(X_train, y_train)
train_predictions = rf.predict(X_train)
test_predictions = rf.predict(X_test)


# In[217]:


train_error = MAE(y_true=y_train, y_pred=train_predictions)
test_error = MAE(y_true=y_test, y_pred=test_predictions)


# In[218]:


print('Model error on seen data: {0:.2f}.'.format(train_error))
print('Model error on unsseen data: {0:.2f}.'.format(test_error))


# In[219]:


#cross validation metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error


# In[220]:


rf = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=1111)
mse= make_scorer(mean_squared_error)


# In[221]:


cv_results = cross_val_score(rf, X, y, cv=5, scoring=mse)
print(cv_results)


# In[222]:


print(MSE(y_test, y_pred))


# In[223]:


#GridSearch cross validation
from sklearn.model_selection import GridSearchCV
SEED=1
rf= RandomForestRegressor(random_state=SEED)


# In[224]:


rf.get_params()


# In[225]:


params_rf = {'n_estimators': [300, 400, 3000],
            'max_depth': [4, 6, 8],
            'min_samples_leaf': [0.1, 0.2],
            'max_features': ['log2', 'sqrt']}


# In[ ]:


rf_cv = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       verbose=1, 
                      n_jobs=-1)
rf_cv.fit(X_train, y_train)


# In[ ]:


print('Best score for this Random Forest Regressor model: {:.3f}'.format(rf_cv.best_score_))


# In[ ]:


best_hyperparams = rf_cv.best_params_
print('Best hyperparameters:\n', best_hyperparams)


# In[ ]:


#evaluating the best model performance
best_model = rf_cv.best_estimator_
y_pred= best_model.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))


# In[ ]:




