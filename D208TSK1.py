#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports 
import numpy as np 
import pandas as pd
from pandas import Series, DataFrame 

#Visualization libraries 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Statistics Packages
import pylab
from pylab import rcParams
import statsmodels.api as sm
import statistics
from scipy import stats 

#Scikit-learn

import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report


#Import chisquare from SciPy.stats

from scipy.stats import chisquare
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt 


# In[2]:


#Load dataset#Load dataset 
df = pd.read_csv('churn_clean.csv')


# In[3]:


#Ignore future warning code 
import warnings 
warnings.filterwarnings('ignore') 


# In[4]:


#Evaluate dataset to identify datatypes and column names
df.info()


# In[5]:


#Shape dataset
df.shape


# In[6]:


#Summary Statistics
df.describe()


# In[7]:


#Renaming survey columns for better descriptions of variables
df.rename(columns= {'Item1':'TimelyResponse', 'Item2': 'Fixes', 
                    'Item3': 'Replacements', 'Item4': 'Reliability', 'Item5': 'Options', 'Item6': 'Respectfulness',
                    'Item7': 'Courteous', 'Item8': 'Listening'}, inplace=True) 


# In[8]:


#Remove less meaningful demographic variables from stats description
df= df.drop (columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State', 'County', 'Zip', 'Lat', 'Area', 'PaymentMethod', 'TimeZone', 'Job', 'Marital'])


# In[9]:


#Remove less meaningful demographic variables from stats description
df= df.drop (columns=['Lng'])


# In[10]:


#New stats description 
df.describe()


# In[11]:


#Null values 
df.isnull().sum()


# In[12]:


#Create Boxplot to check for outliers in continous variables
sns.boxplot('Tenure', data = df)
plt.show()


# In[13]:


sns.boxplot ('MonthlyCharge', data = df)
plt.show()


# In[14]:


sns.boxplot ('Bandwidth_GB_Year', data = df)
plt.show()


# In[15]:


#yes/no dummy variables
df['DummyGender'] = [1 if v == 'Male' else 0 for v in df['Gender']]
df['DummyChurn'] = [1 if v == 'Male' else 0 for v in df['Churn']]
df['DummyTechie'] = [1 if v == 'Male' else 0 for v in df['Techie']]
df['DummyContract'] = [1 if v == 'Male' else 0 for v in df['Contract']]
df['DummyPort_modem'] = [1 if v == 'Male' else 0 for v in df['Port_modem']]
df['DummyTablet'] = [1 if v == 'Male' else 0 for v in df['Tablet']]
df['DummyInternetService'] = [1 if v == 'Male' else 0 for v in df['InternetService']]
df['DummyPhone'] = [1 if v == 'Male' else 0 for v in df['Phone']]
df['DummyMultiple'] = [1 if v == 'Male' else 0 for v in df['Multiple']]
df['DummyOnlineSecurity'] = [1 if v == 'Male' else 0 for v in df['OnlineSecurity']]
df['DummyOnlineBackup'] = [1 if v == 'Male' else 0 for v in df['OnlineBackup']]
df['DummyDeviceProtection'] = [1 if v == 'Male' else 0 for v in df['DeviceProtection']]
df['DummyTechSupport'] = [1 if v == 'Male' else 0 for v in df['TechSupport']]
df['DummyStreamingTV'] = [1 if v == 'Male' else 0 for v in df['StreamingTV']]
df['DummyStreamingMovies'] = [1 if v == 'Male' else 0 for v in df['StreamingMovies']]
df['DummyPaperlessBilling'] = [1 if v == 'Male' else 0 for v in df['PaperlessBilling']]


# In[16]:


#drop original categorical columns from data frame 
df = df.drop(columns=['Gender', 'Churn', 'Techie', 'Contract', 'Port_modem', 'Tablet', 'OnlineSecurity', 'InternetService',
                      'Phone', 'Multiple', 'TechSupport', 'PaperlessBilling', 'OnlineBackup', 'DeviceProtection', 
                      'StreamingTV', 'StreamingMovies'])


# In[17]:


df.describe()


# In[18]:


#List of dataframe columns for next steps
print(df.keys())


# In[19]:


#Move column "Bandwidth_GB_Year to end of dataset 
df = df[[ 'Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure',
         'Tenure', 'MonthlyCharge', 'TimelyResponse', 'Fixes', 'Replacements', 'Reliability', 'Options', 
         'Respectfulness', 'Courteous', 'Listening', 'DummyGender', 'DummyChurn', 'DummyTechie', 'DummyContract', 
         'DummyPort_modem', 'DummyTablet', 'DummyInternetService', 'DummyPhone', 'DummyMultiple', 'DummyOnlineSecurity',
         'DummyOnlineBackup', 'DummyDeviceProtection', 'DummyTechSupport', 'DummyStreamingTV', 
         'DummyPaperlessBilling', 'Bandwidth_GB_Year']]


# In[20]:


#Review Clean Dataset
df.shape


# In[21]:


#List of dataframe columns for next steps
print(df.keys())


# In[22]:


#Univariate Statistics Histogram 
df[['Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure', 'Tenure', 
    'MonthlyCharge', 'Bandwidth_GB_Year']].hist()
plt.savefig('churn_pyplot.jpg')
plt.tight_layout()


# In[23]:


#Bivariate Statistics 
sns.scatterplot(x=df['Children'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[24]:


#Bivariate Statistics 
sns.scatterplot(x=df['Age'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[25]:


#Bivariate Statistics 
sns.scatterplot(x=df['Income'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[26]:


#Bivariate Statistics 
sns.scatterplot(x=df['Outage_sec_perweek'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[27]:


#Bivariate Statistics 
sns.scatterplot(x=df['Email'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[28]:


#Bivariate Statistics 
sns.scatterplot(x=df['Contacts'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[29]:


#Bivariate Statistics 
sns.scatterplot(x=df['Yearly_equip_failure'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[30]:


#Bivariate Statistics 
sns.scatterplot(x=df['Tenure'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[31]:


#Bivariate Statistics 
sns.scatterplot(x=df['MonthlyCharge'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[32]:


#Bivariate Statistics 
sns.scatterplot(x=df['TimelyResponse'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[33]:


#Bivariate Statistics 
sns.scatterplot(x=df['Fixes'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[34]:


#Bivariate Statistics 
sns.scatterplot(x=df['DummyTechie'], y= df ['Bandwidth_GB_Year'], color='red')
plt.show(); 


# In[35]:


#Extract Clean dataset
df.to_csv('churn_prepared.csv')


# In[36]:


#List of dataframe columns for next steps
print(df.keys())


# In[37]:


#Initial Multiple Regression Model using continous variables
df['intercept'] = 1
lm_bandwidth = sm.OLS(df['Bandwidth_GB_Year'], df[['Children', 'Age', 'Income', 
                                                          'Outage_sec_perweek', 'Email', 'Contacts', 
                                                          'Yearly_equip_failure', 'Tenure', 'MonthlyCharge',
                                                          'TimelyResponse', 'Fixes', 'Replacements', 'Reliability',
                                                          'Respectfulness', 'Options', 'Courteous', 
                                                          'Listening','intercept']]).fit()
print(lm_bandwidth.summary())


# In[38]:


#List of dataframe columns for next steps
print(df.keys())


# In[39]:


#Multiple Regression Model using categorical dummy variables
df['intercept'] = 1
lm_bandwidth = sm.OLS(df['Bandwidth_GB_Year'], df[['Children', 'Age', 'Income', 
                                                          'Outage_sec_perweek', 'Email', 'Contacts', 
                                                          'Yearly_equip_failure', 'Tenure', 'DummyContract',
                                                          'DummyTechie', 'DummyPort_modem', 'DummyTablet',
                                                          'DummyInternetService', 'DummyPhone', 'DummyMultiple',
                                                          'DummyOnlineSecurity', 'DummyOnlineBackup', 
                                                          'DummyStreamingTV', 'DummyPaperlessBilling',
                                                          'MonthlyCharge','TimelyResponse', 'Fixes', 
                                                          'Reliability', 'Replacements', 'Options',
                                                          'Respectfulness', 'Courteous', 'Listening', 
                                                          'intercept']]).fit()
print(lm_bandwidth.summary())


# In[40]:


#Create Dataframe for Heatmap bivariate Analysis of Correlation
churn_bivariate = df[['Bandwidth_GB_Year', 'Children', 'Age', 'Income', 'Outage_sec_perweek', 'Yearly_equip_failure', 
                      'DummyTechie', 'DummyContract', 'DummyPort_modem', 'DummyTablet', 'DummyInternetService',
                      'DummyPhone', 'DummyMultiple', 'DummyOnlineSecurity', 'DummyOnlineBackup', 
                      'DummyDeviceProtection', 'DummyTechSupport', 'DummyStreamingTV', 'DummyPaperlessBilling', 
                      'Email', 'Contacts', 'Tenure', 'MonthlyCharge', 'TimelyResponse', 'Fixes', 'Replacements',
                      'Reliability', 'Options', 'Respectfulness', 'Courteous', 'Listening']]


# In[41]:


#Run Heatmap Utilizing Seaborn
sns.heatmap(churn_bivariate.corr(), annot=False)
plt.show()


# In[42]:


#Create Dataframe for Heatmap Bivariate Analysis of Correlation on Purple or Darker Variables in Previous Heatmap
churn_bivariate = df[['Bandwidth_GB_Year', 'Children', 'Tenure','TimelyResponse', 'Fixes', 'Replacements',
                       'Respectfulness', 'Courteous', 'Listening']]


# In[43]:


#Run Heatmap Utilizing Seaborn
sns.heatmap(churn_bivariate.corr(), annot=True)
plt.show()


# In[47]:


#Reduced OLS Multiple Regression 
lm_bandwidth_reduced = sm.OLS(df['Bandwidth_GB_Year'], 
                               df[['Children', 'Tenure', 'Fixes', 'Replacements', 'intercept']]).fit()
                                   
print(lm_bandwidth_reduced.summary())


# In[45]:


#Residual Plot 
import seaborn as sns
sns.histplot(lm_bandwidth_reduced.resid); 


# In[750]:


#Residual Plot 
from scipy import stats 
mu, std = stats.norm.fit(lm_bandwidth_reduced.resid)
mu, std


# In[751]:


import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
# plot the residuals
sns.histplot(x=lm_bandwidth_reduced.resid, ax=ax, stat="density", linewidth=0, kde=True)
ax.set(title="Distribution of residuals", xlabel="residual")

# plot corresponding normal curve
xmin, xmax = plt.xlim() # the maximum x values from the histogram above
x = np.linspace(xmin, xmax, 100) # generate some x values
p = stats.norm.pdf(x, mu, std) # calculate the y values for the normal curve
sns.lineplot(x=x, y=p, color="orange", ax=ax)
plt.show()


# In[752]:


sns.boxplot(x=lm_bandwidth_reduced.resid, showmeans=True);


# In[753]:


sm.qqplot(lm_bandwidth_reduced.resid, line='s');


# In[754]:


sm.graphics.plot_fit(lm_bandwidth_reduced,1, vlines=False);


# In[755]:


lm_bandwidth_reduced.fittedvalues


# In[756]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

data = sm.datasets.spector.load(as_pandas=False)
X = data.exog
y = data.endog

# fit the model
model = sm.OLS(y, sm.add_constant(X, prepend=False))
fit = model.fit()


# In[757]:


# fit the model
model = sm.OLS(y, sm.add_constant(X, prepend=False))
fit = model.fit()



# In[758]:


# compute the residuals and other metrics
influence = OLSInfluence(fit)


# In[ ]:





# In[ ]:





# In[ ]:




