#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Logistic Regression for Predictive Modeling
# import libraries
import numpy as np
import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.core.display import HTML
from IPython.display import display
import os

import matplotlib.pyplot as plt
plt.rc("font", size=14)
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn import metrics


import seaborn as sns

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from tkinter import *

#Ignore future warning code 
import warnings 
warnings.filterwarnings('ignore') 


# In[2]:


#Helper Functions
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def custom_corr_matrix(df, title):
    fig = plt.figure(figsize=(30, 30))
    sns.set(font_scale=1.0)
    sns.heatmap(data=df.corr().round(1), annot=True,annot_kws={'size':30})
    print(get_top_abs_correlations(df))
    #plt.savefig('output/' + COURSE + '/fig_corr_matrix_' + title + '.png', facecolor='w') 
    plt.show()
    
def plot_histogram(c):
    df_yes = df[df.Churn_Yes==1][c]
    df_no = df[df.Churn_Yes==0][c]
    yes_mean = df_yes.mean();
    no_mean = df_no.mean(); 
    fig,ax = plt.subplots(figsize=(6,6))
    ax.hist([df_yes,df_no], bins=5, stacked=True)
    ax.legend(['Churn - Yes','Churn - No'])
    ymin, ymax = ax.get_ylim();
    xmin, xmax = ax.get_xlim()
    ax.axvline(yes_mean, color='blue', lw=2) # yes mean
    ax.axvline(no_mean, color='orangered', lw=2) # no mean
    ax.text((xmax-xmin)/2,
            (ymax-ymin)/2,
            'Delta:\n' + str(round(abs(yes_mean - no_mean),2)),
            bbox={'facecolor':'white'})
    plt.title('Histogram with target overlay by ' + str(c))
    plt.xlabel(c); 
    plt.ylabel('# Churn');
    plt.show();
    
# helper function to plot grouped bar plot
def plot_stacked(c):
    df.groupby([c,target]).size().unstack().plot(kind='bar', stacked=True)


# In[3]:


#constants
COURSE = 'd208' # name of course to be added to filename of generated figures and tables.
target = 'Churn' # this is the column name of the primary research column


# In[4]:


#read csv
df = pd.read_csv('churn_clean.csv')


# In[5]:


#evaluate for missing values
missing = df[df.columns[df.isna().any()]].columns
df_missing = df[missing]
print(df_missing.info())


# In[6]:


#evaluate for duplicates
df.duplicated().any()


# In[7]:


df.shape


# In[8]:


# drop unwanted data
cols_to_be_removed = ['City','County','Zip','Job','TimeZone', 'State', 
            'Lat', 'Lng', 'UID', 'Customer_id','Interaction', 'CaseOrder',
            'Item1','Item2','Item3','Item4','Item5','Item6','Item7','Item8']

# print list of dropped data
print('data to be removed: {}'.format(cols_to_be_removed))

# loop through list, if in current df, drop col
for c in cols_to_be_removed:
    if c in df.columns:
        df.drop(columns = c, inplace=True)
        print('Data named [{}] has been removed.'.format(c))


# In[9]:


df.shape


# In[10]:


# print out and describe input variables
for idx, c in enumerate(df.loc[:, df.columns != target]):
    if df.dtypes[c] == "object":
        print('\n{}. {} is categorical: {}.'.format(idx+1,c,df[c].unique()))
        #for idx,name in enumerate(df[c].value_counts().index.tolist()):
        #    print('\t{:<20}:{:>6}'.format(name,df[c].value_counts()[idx]))
        #print('{}'.format(df[c].describe()))
    else:
        print('\n{}. {} is numerical.'.format(idx+1, c))
        #print('{}'.format(df[c].describe().round(3)))
        #groups = df.groupby([target, pd.cut(df[c], bins=4)])
        #print(groups.size().unstack().T)


# In[11]:


# print out and describe target variable
for idx, c in enumerate(df.loc[:, df.columns == target]):
    if df.dtypes[c] == "object":
        print('\n{}. {} is categorical: {}.'.format(idx+1,c,df[c].unique()))
        for idx,name in enumerate(df[c].value_counts().index.tolist()):
            print('\t{:<8}:{:>6}'.format(name,df[c].value_counts()[idx]))
    else:
        print('\n{}. {} is numerical.'.format(idx+1, c))


# In[12]:


# variable for numeric data
num_cols = df.select_dtypes(include="number").columns
print(num_cols)


# In[13]:


# variable for categorical data
cat_cols = df.select_dtypes(include="object").columns
print(cat_cols)


# In[14]:


print(df[target].value_counts())
sns.countplot(x=target, data=df, palette='hls')
plt.show()


# In[15]:


# calculate balance
count_no_churn = len(df[df[target]=='No'])
count_churn = len(df[df[target]=='Yes'])
pct_of_no_churn = count_no_churn/(count_no_churn+count_churn)
pct_of_churn = count_churn/(count_no_churn+count_churn)
print('% of customers that did not churn: {:.1%}'.format(pct_of_no_churn ))
print('% of customers that did churn: {:.1%}'.format(pct_of_churn ))


# In[16]:


# describe numerical mean data compared to target
df.groupby(target).mean().round(2).T


# In[17]:


# plot categorical data - before it gets converted
fig = plt.figure(figsize=(10, 20))

for i, col in enumerate(cat_cols):
    if col != target:
        plt.subplot(10, 3, i+1)
        ax = sns.countplot(y=col, data=df)
        fig.tight_layout(h_pad=4, w_pad=4)

plt.title('Categorical Data')
plt.show()


# In[18]:


# print out mean values of numeric data for a given variable 
for c in cat_cols:
    if c != target:
        print('\n\n======================================')
        print('\t{}'.format(c.upper()))
        print('======================================')
        print(df.groupby(c).mean().round(2).T)


# In[19]:


# plot each variable vs. target overlay
for c in cat_cols:
    if c != target:
        plot_stacked(c)


# In[20]:


# convert categorical data to dummy data
for c in cat_cols:
    if c in df.columns:
        df = pd.get_dummies(df, columns=[c], drop_first=True)
pred_vars = df.select_dtypes(include="uint8").columns.tolist()
print(pred_vars)


# In[21]:


df.shape


# In[22]:


# reset the global target variable using its dummy variable
target = 'Churn_Yes'


# In[23]:


# describe numeric data
df[num_cols].describe().round(3).T


# In[24]:


#data types with new dummy variables
df.info()


# In[25]:


# histogram plot numeric data
fig = plt.figure(figsize=(10, 20))
ax = df[num_cols].hist(bins = 15, figsize=(15,15))
plt.title('Numeric Data')
fig.tight_layout(h_pad=5, w_pad=5)
plt.show()


# In[26]:


# create histogram with target overlay
plot_histogram('MonthlyCharge')


# In[27]:


# create histogram with target overlay
plot_histogram('Tenure')


# In[28]:


# create histogram with target overlay
plot_histogram('Bandwidth_GB_Year')


# In[29]:


# create histogram with target overlay
plot_histogram('Age')


# In[30]:


# create histogram with target overlay
plot_histogram('Income')


# In[31]:


# create histogram with target overlay
plot_histogram('Outage_sec_perweek')


# In[32]:


# create histogram with target overlay
plot_histogram('Email')


# In[33]:


# create histogram with target overlay
plot_histogram('Contacts')


# In[34]:


# create histogram with target overlay
plot_histogram('Yearly_equip_failure')


# In[35]:


# create histogram with target overlay
plot_histogram('Population')


# In[36]:


# create histogram with target overlay
plot_histogram('Children')


# In[37]:


# Provide copy of the prepared data set.
final_data = 'd208_task2_final_data.csv'
df.to_csv(final_data, index=False, header=True)


# In[38]:


#balance data using SMOTE oversample - Takes data from 70/30 to 50/50
X = df.loc[:, df.columns != 'Churn_Yes']
y = df.loc[:, df.columns == 'Churn_Yes']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['Churn_Yes'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no churn in oversampled data",len(os_data_y[os_data_y['Churn_Yes']==0]))
print("Number of churn",len(os_data_y[os_data_y['Churn_Yes']==1]))
print("Proportion of no churn data in oversampled data is ",len(os_data_y[os_data_y['Churn_Yes']==0])/len(os_data_X))
print("Proportion of churn data in oversampled data is ",len(os_data_y[os_data_y['Churn_Yes']==1])/len(os_data_X))


# In[39]:


# RFE feature reduction
data_final_vars=df.columns.values.tolist()
y=[target]
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, step = 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
features =[]
print('The following features are selected:')
for i in range(os_data_X.shape[1]):
    if rfe.support_[i] == True:
        features.append(os_data_X.columns[i])
        print('Column: %d, Rank: %.3f, Feature: %s' % 
          (i, rfe.ranking_[i],
           os_data_X.columns[i]))


# In[40]:


# initial model
X=os_data_X[features] # from RFE above
Xc = sm.add_constant(X) # reset
y=os_data_y[target]
logit_model=sm.Logit(y,Xc)
result=logit_model.fit()
print(result.summary2())


# In[41]:


# confustion matrix for initial model
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
lgr = LogisticRegression()
lgr.fit(X_train, y_train)
predicted = lgr.predict(X_test)
expected = y_test
confusion = pd.DataFrame(confusion_matrix(y_true=expected, y_pred=predicted),
                    index=range(2),columns=range(2))
axes = sns.heatmap(confusion, annot=True,cmap='nipy_spectral_r', fmt='g')


# In[42]:


#calculate number and percent of predictions
correct = sum(np.diagonal(confusion)) # on diag
total = confusion.values.sum()
incorrect = total - correct # off diag
print('Correct predictions on diagonal: {} ({:.0%})'.format( correct, correct / total ))
print('Incorrect predictions off diagonal: {} ({:.0%})'.format( incorrect, incorrect / total )) 


# In[43]:


#top variables with high correlation
#custom_corr_matrix(X,'Model_2')
get_top_abs_correlations(X, 20)


# In[45]:


# update model
features.remove('Contract_Two Year') #high correlation
features.remove('Outage_sec_perweek') #p-value over 0.05
features.remove('Contacts')  #p-value over 0.05
features.remove('Port_modem_Yes') #p-value over 0.05
features.remove('PaperlessBilling_Yes') #p-value over 0.05
features.remove('InternetService_Fiber Optic') # high collinearity
X=os_data_X[features]
y=os_data_y[target]
Xc = sm.add_constant(X) # reset
logit_model=sm.Logit(y,Xc)
result=logit_model.fit()
print(result.summary2())


# In[46]:


# confustion matrix to identify percentage of correct predictions
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
lgr = LogisticRegression()
lgr.fit(X_train, y_train)
predicted = lgr.predict(X_test)
expected = y_test
confusion = pd.DataFrame(confusion_matrix(y_true=expected, y_pred=predicted),
                    index=range(2),columns=range(2))
axes = sns.heatmap(confusion, annot=True,cmap='nipy_spectral_r', fmt='g')


# In[47]:


# calculate number and percent of predictions
correct = sum(np.diagonal(confusion)) # on diag
total = confusion.values.sum()
incorrect = total - correct # off diag
print('Correct predictions on diagonal: {} ({:.0%})'.format( correct, correct / total ))
print('Incorrect predictions off diagonal: {} ({:.0%})'.format( incorrect, incorrect / total )) 


# In[48]:


# classification report for f1 score
print(classification_report(expected, predicted))


# In[49]:


# plot ROC Curve
logit_roc_auc = roc_auc_score(y_test, lgr.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lgr.predict_proba(X_test)
[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' %logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[50]:


# equation of the regression line/plane
print('Logit: {:.2f}'.format(logit_roc_auc))
equation = result.summary2().tables[1]
print('Estimate [{}] as L = '.format(result.summary2().tables[0][1][1]))
for i in equation.itertuples():
    print('   {:+.3f} x ( {} ) '.format(i[1],i[0]))

