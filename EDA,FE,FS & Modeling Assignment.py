#!/usr/bin/env python
# coding: utf-8

# # Building a Machine Learning model to predict the date of payment of an invoice.
# 

# In[5]:


import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
import seaborn as sns
import math


# In[6]:


#Importing Data Set--> csv.csv
df = pd.read_csv('csv.csv')
df.head()


# # Data Pre-Processing and Filtering

# In[7]:


df.info()


# In[8]:


#Converting all date (present in float and object ) to datetime format
df['due_in_date'] = pd.to_datetime(df['due_in_date'], format='%Y%m%d')
df['document_create_date.1'] = pd.to_datetime(df['document_create_date.1'], format='%Y%m%d')
df['clear_date'] = pd.to_datetime(df['clear_date'])
df['posting_date'] = pd.to_datetime(df['posting_date'])
df['baseline_create_date'] = pd.to_datetime(df['baseline_create_date'], format='%Y%m%d')


# In[9]:


#Saving a copy of test data for later purpose.
test_final=df.loc[df["clear_date"].isnull()].reset_index(drop=True)


# In[10]:


#Resetting index.
test_final.reset_index(inplace = True)
test_final.drop(['index'], axis=1, inplace=True)


# In[11]:


df.info()


# In[12]:


#Checking for the null values in invoice column.
df['invoice_id'].isna().sum()


# In[13]:


#Dropping
df.drop('document_create_date',axis = 1, inplace = True)


# In[14]:


df.info()


# In[15]:


df['delay'] = (df['clear_date']-df['due_in_date'])


# In[16]:


df.head()


# In[17]:


df['delay'] = df['delay'].dt.days


# In[18]:


df.head()


# In[19]:


df['delay'].describe()


# In[20]:


df.rename(columns={'buisness_year':'business_year'},inplace=True)


# In[21]:


df.info()


# In[22]:


df.groupby('posting_id').count()


# In[23]:


df.groupby('document type').count()


# In[24]:


#Removing Constant Columns ->> Columns which have the same value in the enitre dataset
unique_cols =  [x for x in df.columns if df[x].nunique()==1] 
print(unique_cols)
df.drop(unique_cols,axis=1,inplace=True)
df.columns


# In[25]:


#For Removing Null Column
df['area_business'].isna().count()


# In[21]:


df.drop('area_business',axis = 1, inplace = True)


# In[22]:


#Checking for same row values in ('doc_id' and 'invoice_id')
print((df['doc_id'] - df['invoice_id']).sum())


# In[23]:


#Checking for same row values in ('posting_date' and 'doc_create_date')
print((df['posting_date'] - df['document_create_date.1']).sum())


# In[24]:


df.drop('doc_id',axis = 1, inplace = True)
df.drop('posting_date',axis = 1, inplace = True)


# In[25]:


df.head()


# In[26]:


#splitting the dataset into train data and test data
test=df.loc[df["clear_date"].isnull()].reset_index(drop=True)
train=df.loc[df["clear_date"].notnull()].reset_index(drop=True)


# In[27]:


print(train.shape)


# In[28]:


# Removing anomalies for train set
train=train[((train['document_create_date.1']<=train['baseline_create_date']) & (train['baseline_create_date']<=train['due_in_date']))|((train['document_create_date.1']<=train['baseline_create_date']) & (train['baseline_create_date']<=train['clear_date']))]

# Removing anomalies for test set
test=test[((test['document_create_date.1']<=test['baseline_create_date']) & (test['baseline_create_date']<=test['due_in_date']))|(test['document_create_date.1']<=test['baseline_create_date'])]


# In[29]:


# Removing anomalies for test_final set
test_final=test_final[((test_final['document_create_date.1']<=test_final['baseline_create_date']) & (test_final['baseline_create_date']<=test_final['due_in_date']))|(test_final['document_create_date.1']<=test_final['baseline_create_date'])]


# In[30]:


print(train.shape)


# In[31]:


train


# In[32]:


train.isnull().sum()
#No need to for null imputation


# In[33]:


#Removing Constant Columns ->> Columns which have the same value in the enitre dataset
train.drop(['isOpen'], axis=1, inplace=True)
test.drop(['isOpen'], axis=1, inplace=True)


# In[34]:


#Sorting the train_set according to 'document_create_date.1' to slice properly into val1 and val2.
train.sort_values(by=['document_create_date.1'], inplace=True)


# In[35]:


train.shape


# In[36]:


#Box Plot to check the outliers.
train.boxplot(column =['total_open_amount'], grid = False)
#There are lot of outliers , nothing should be dropped, if outliers dropped then it our lots of data will be lost.


# In[37]:


#Slicing the data set to validation_set1 and validation_set2 and train_set
val1 = train.iloc[30710:37290] #70-->85
val2 = train.iloc[37290:] #85-->100
train = train.iloc[:30710] #Upto 70


# In[38]:


val1.shape


# In[39]:


val2.shape


# In[40]:


train.shape


# In[41]:


train


# In[42]:


#Resetting index according to sorting of 'document_create_date.1'.
train.reset_index(inplace=True)
val1.reset_index(inplace=True)
val2.reset_index(inplace=True)


# # EDA and Feature Engineering

# In[43]:


# target encoding 'name_customer' column

from collections import defaultdict

name = {}
name = (train.groupby('name_customer')['delay'].mean()).to_dict()

def delay_mean():
    return train['delay'].mean()

cust_name = {}
cust_name = defaultdict(delay_mean)

for i,j in name.items():
    cust_name[i] = j
    
train['name_customer'] = train['name_customer'].map(cust_name)
val1['name_customer'] = val1['name_customer'].map(cust_name)
val2['name_customer'] = val2['name_customer'].map(cust_name)
test['name_customer'] = test['name_customer'].map(cust_name)


# In[44]:


# target encoding 'cust_number' column

from collections import defaultdict

name = {}
name = (train.groupby('cust_number')['delay'].mean()).to_dict()

def delay_mean():
    return train['delay'].mean()

cust_num = {}
cust_num = defaultdict(delay_mean)

for i,j in name.items():
    cust_num[i] = j
    
train['cust_number'] = train['cust_number'].map(cust_num)
val1['cust_number'] = val1['cust_number'].map(cust_num)
val2['cust_number'] = val2['cust_number'].map(cust_num)
test['cust_number'] = test['cust_number'].map(cust_num)


# In[45]:


# one hot encoding cust_payment_terms column
code = {}
c=0
for i in train['cust_payment_terms']:
    if i not in code:
        code[i] = c
        c += 1
    
train['cust_payment_terms'] = train['cust_payment_terms'].map(code)
val1['cust_payment_terms'] = val1['cust_payment_terms'].map(code)
val2['cust_payment_terms'] = val2['cust_payment_terms'].map(code)
test['cust_payment_terms'] = test['cust_payment_terms'].map(code)


# In[46]:


train.head()


# In[49]:


train.drop('index',axis = 1, inplace = True)
val1.drop('index',axis = 1, inplace = True)
val2.drop('index',axis = 1, inplace = True)


# In[50]:


#Converting CAD to USD, for dropping the invoice currency column.
from forex_python.converter import CurrencyRates 

c = CurrencyRates()
train.loc[(train.invoice_currency != 'USD'), ['total_open_amount']] *= c.get_rate('CAD', 'USD')
test.loc[(test.invoice_currency != 'USD'), ['total_open_amount']] *= c.get_rate('CAD', 'USD')
val1.loc[(val1.invoice_currency != 'USD'), ['total_open_amount']] *= c.get_rate('CAD', 'USD')
val2.loc[(val2.invoice_currency != 'USD'), ['total_open_amount']] *= c.get_rate('CAD', 'USD')


# In[51]:


# one hot encoding invoice_currency column

curr = {}
c=0
for i in train['invoice_currency']:
    if i not in curr:
        curr[i] = c
        c += 1

train['invoice_currency'] = train['invoice_currency'].map(curr)
val1['invoice_currency'] = val1['invoice_currency'].map(curr)
val2['invoice_currency'] = val2['invoice_currency'].map(curr)
test['invoice_currency'] = test['invoice_currency'].map(curr)


# In[52]:


# one hot encoding business_code column
code = {}
c=0
for i in train['business_code']:
    if i not in code:
        code[i] = c
        c += 1
    
train['business_code'] = train['business_code'].map(code)
val1['business_code'] = val1['business_code'].map(code)
val2['business_code'] = val2['business_code'].map(code)
test['business_code'] = test['business_code'].map(code)


# In[53]:


#Dropping 'invoice_currency'
train.drop('invoice_currency',axis = 1, inplace = True)
test.drop('invoice_currency',axis = 1, inplace = True)
val1.drop('invoice_currency',axis = 1, inplace = True)
val2.drop('invoice_currency',axis = 1, inplace = True)


# In[54]:


train.head()


# In[55]:


# calculating difference between the date columns

train['due_create'] = (train['due_in_date'] - train['document_create_date.1']).dt.days
train['create_base'] = (train['document_create_date.1'] - train['baseline_create_date']).dt.days
train['due_base'] = (train['due_in_date'] - train['baseline_create_date']).dt.days

val1['due_create'] = (val1['due_in_date'] - val1['document_create_date.1']).dt.days
val1['create_base'] = (val1['document_create_date.1'] - val1['baseline_create_date']).dt.days
val1['due_base'] = (val1['due_in_date'] - val1['baseline_create_date']).dt.days

val2['due_create'] = (val2['due_in_date'] - val2['document_create_date.1']).dt.days
val2['create_base'] = (val2['document_create_date.1'] - val2['baseline_create_date']).dt.days
val2['due_base'] = (val2['due_in_date'] - val2['baseline_create_date']).dt.days

test['due_create'] = (test['due_in_date'] - test['document_create_date.1']).dt.days
test['create_base'] = (test['document_create_date.1'] - test['baseline_create_date']).dt.days
test['due_base'] = (test['due_in_date'] - test['baseline_create_date']).dt.days


# In[56]:


# Extracting month from each of the date columns

train['doc_create_month'] = train['document_create_date.1'].dt.month
train['due_month'] = train['due_in_date'].dt.month
train['base_create_month'] = train['baseline_create_date'].dt.month

val1['doc_create_month'] = val1['document_create_date.1'].dt.month
val1['due_month'] = val1['due_in_date'].dt.month
val1['base_create_month'] = val1['baseline_create_date'].dt.month

val2['doc_create_month'] = val2['document_create_date.1'].dt.month
val2['due_month'] = val2['due_in_date'].dt.month
val2['base_create_month'] = val2['baseline_create_date'].dt.month

test['doc_create_month'] = test['document_create_date.1'].dt.month
test['due_month'] = test['due_in_date'].dt.month
test['base_create_month'] = test['baseline_create_date'].dt.month


# In[57]:


# keeping the invoice_id and due_in_date columns of the train, validation and test set in a separate dataframe

f_train = pd.DataFrame(columns=['invoice_id', 'due_in_date'])
f_val1 = pd.DataFrame(columns=['invoice_id', 'due_in_date'])
f_val2 = pd.DataFrame(columns=['invoice_id', 'due_in_date'])
f_test = pd.DataFrame(columns=['invoice_id', 'due_in_date'])

f_train['invoice_id'] = train['invoice_id']
f_val1['invoice_id'] = val1['invoice_id']
f_val2['invoice_id'] = val2['invoice_id']
f_test['invoice_id'] = test['invoice_id']

f_train['due_in_date'] = train['due_in_date']
f_val1['due_in_date'] = val1['due_in_date']
f_val2['due_in_date'] = val2['due_in_date']
f_test['due_in_date'] = test['due_in_date']


# In[58]:


# Extracting year from each of the date columns

train['doc_create_year'] = train['document_create_date.1'].dt.year
train['due_year'] = train['due_in_date'].dt.year
train['base_create_year'] = train['baseline_create_date'].dt.year

val1['doc_create_year'] = val1['document_create_date.1'].dt.year
val1['due_year'] = val1['due_in_date'].dt.year
val1['base_create_year'] = val1['baseline_create_date'].dt.year

val2['doc_create_year'] = val2['document_create_date.1'].dt.year
val2['due_year'] = val2['due_in_date'].dt.year
val2['base_create_year'] = val2['baseline_create_date'].dt.year

test['doc_create_year'] = test['document_create_date.1'].dt.year
test['due_year'] = test['due_in_date'].dt.year
test['base_create_year'] = test['baseline_create_date'].dt.year


# In[59]:


train.corr()['delay'].sort_values(ascending = False)


# In[60]:


#visualisation using heat_map
corr=train.corr()
plt.figure(figsize=(14,8))
sns.heatmap(corr,annot=True)


# In[61]:


#Dropping below columns after extracting the useful features.
train.drop(['business_year','cust_number','invoice_id','doc_create_year','due_year','base_create_year','clear_date','due_in_date','document_create_date.1','baseline_create_date'],axis = 1, inplace = True)
test.drop(['business_year','cust_number','invoice_id','doc_create_year','due_year','base_create_year','clear_date','due_in_date','document_create_date.1','baseline_create_date','delay'],axis = 1, inplace = True)
val1.drop(['business_year','cust_number','invoice_id','doc_create_year','due_year','base_create_year','clear_date','due_in_date','document_create_date.1','baseline_create_date'],axis = 1, inplace = True)
val2.drop(['business_year','cust_number','invoice_id','doc_create_year','due_year','base_create_year','clear_date','due_in_date','document_create_date.1','baseline_create_date'],axis = 1, inplace = True)


# In[62]:


train.isna().sum()


# In[63]:


val1.isna().sum()


# In[64]:


val2.isna().sum()


# In[66]:


#Dropping the null values
val1.dropna(inplace=True)
val2.dropna(inplace=True)


# In[67]:


train.shape


# In[68]:


#visualisation using heatmap
corr=train.corr()
plt.figure(figsize=(14,8))
sns.heatmap(corr,annot=True)


# # Feature Selection

# In[69]:


# Displaying important features using Random Forest Regressor

def tree_based_feature_importance(x_train,y_train):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=120)
    model.fit(x_train, y_train)
    importances = model.feature_importances_
    final_df = pd.DataFrame({"Features": x_train.columns, "Importances":importances})
    final_df.set_index('Importances')
    final_df = final_df.sort_values('Importances',ascending=False)
    pd.Series(model.feature_importances_, index=x_train.columns).nlargest(10).plot(kind='barh',color='cyan')  
    return final_df
X_train = train[['business_code', 'name_customer', 'total_open_amount', 'cust_payment_terms', 'due_create', 'create_base', 'due_base', 'doc_create_month', 'due_month', 'base_create_month']]
Y_train = train[['delay']]
feature_importance=tree_based_feature_importance(X_train,Y_train)


# In[70]:


train.corr()['delay'].sort_values(ascending = False)


# In[71]:


x_train = train.loc[:, ['business_code', 'due_create', 'name_customer', 'due_base']] 
y_train = train['delay']

x_val1 = val1.loc[:, ['business_code', 'due_create', 'name_customer', 'due_base']] 
y_val1 = val1['delay']

x_val2 = val2.loc[:, ['business_code', 'due_create', 'name_customer', 'due_base']]
y_val2 = val2['delay']

x_test = test.loc[:, ['business_code', 'due_create', 'name_customer', 'due_base']]


# In[72]:


# Feature Scaling

from sklearn.preprocessing import StandardScaler

scalerx = StandardScaler().fit(x_train)

x_train = scalerx.transform(x_train)

x_val1 = scalerx.transform(x_val1)

x_val2 = scalerx.transform(x_val2)

x_test = scalerx.transform(x_test)


# # Modeling

# In[73]:


# Modeling & Accuracy Metrics
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# # Fitting and checking model accuracy with validation set 1

# In[74]:


# Defining Lists to Store in the Results and Names of Algorithms
MSE_Score = []
R2_Score = []
Algorithm = []


# In[75]:


# Fitting Simple Linear Regression to the Training Set
Algorithm.append('Linear Regression')
clf = LinearRegression()
clf.fit(x_train, y_train)

# Predicting the val1 Results
predicted = clf.predict(x_val1)


# In[76]:


# Appending the Scores For Visualisation at a Later Part
MSE_Score.append(mean_squared_error(y_val1, predicted))
R2_Score.append(r2_score(y_val1, predicted))


# In[77]:



print("RMSE: ", math.sqrt(mean_squared_error(y_val1, predicted)))
print("Score: ", (clf.score(x_val1, y_val1))*100, "%")


# In[78]:


# Fitting SVR to the Training Set
Algorithm.append('Support Vector Regression')
clf = SVR()
clf.fit(x_train, y_train)

# Predicting the val1 Results
predicted = clf.predict(x_val1)


# In[79]:


# Appending the Scores For Visualisation at a Later Part
MSE_Score.append(mean_squared_error(y_val1, predicted))
R2_Score.append(r2_score(y_val1, predicted))


# In[80]:


print("RMSE: ", math.sqrt(mean_squared_error(y_val1, predicted)))
print("Score: ", (clf.score(x_val1, y_val1))*100, "%")


# In[81]:


# Fitting Decision Tree to the Training Set
Algorithm.append('Decision Tree Regressor')
clf = DecisionTreeRegressor()
clf.fit(x_train, y_train)

# Predicting the val1 Results
predicted = clf.predict(x_val1)


# In[82]:


MSE_Score.append(mean_squared_error(y_val1, predicted))
R2_Score.append(r2_score(y_val1, predicted))


# In[83]:


print("RMSE: ", math.sqrt(mean_squared_error(y_val1, predicted)))
print("Score: ", (clf.score(x_val1, y_val1))*100, "%")


# In[84]:


# Fitting Random Forest Regressor Tree to the Training Set
Algorithm.append('Random Forest Regressor')
clf = RandomForestRegressor()
clf.fit(x_train, y_train)

# Predicting the val1 Results
predicted = clf.predict(x_val1)


# In[85]:


# Appending the Scores For Visualisation at a Later Part
MSE_Score.append(mean_squared_error(y_val1, predicted))
R2_Score.append(r2_score(y_val1, predicted))


# In[86]:


print("RMSE: ", math.sqrt(mean_squared_error(y_val1, predicted)))
print("Score: ", (clf.score(x_val1, y_val1))*100, "%")


# In[87]:


# Fitting XGBoost Regressor to the Training Set
Algorithm.append('XGB Regressor')
clf = xgb.XGBRegressor()
clf.fit(x_train, y_train)

# Predicting the val1 Results
predicted = clf.predict(x_val1)


# In[88]:


# Appending the Scores For Visualisation at a Later Part
MSE_Score.append(mean_squared_error(y_val1, predicted))
R2_Score.append(r2_score(y_val1, predicted))


# In[89]:


print("RMSE: ", math.sqrt(mean_squared_error(y_val1, predicted)))
print("Score: ", (clf.score(x_val1, y_val1))*100, "%")


# In[90]:


# Just Combining the Lists into a DataFrame for a Better Visualisation
Comparison = pd.DataFrame(list(zip(Algorithm, MSE_Score, R2_Score)), columns = ['Algorithm', 'MSE_Score', 'R2_Score'])


# In[91]:


Comparison


# # Fitting and checking model accuracy with validation set 2

# In[92]:


# Defining Lists to Store in the Results and Names of Algorithms
MSE_Score = []
R2_Score = []
Algorithm = []


# In[93]:


# Fitting Simple Linear Regression to the Training Set
Algorithm.append('Linear Regression')
clf = LinearRegression()
clf.fit(x_train, y_train)

# Predicting the val2 Results
predicted = clf.predict(x_val2)


# In[94]:


# Appending the Scores For Visualisation at a Later Part
MSE_Score.append(mean_squared_error(y_val2, predicted))
R2_Score.append(r2_score(y_val2, predicted))


# In[95]:



print("RMSE: ", math.sqrt(mean_squared_error(y_val2, predicted)))
print("Score: ", (clf.score(x_val2, y_val2))*100, "%")


# In[96]:


# Fitting SVR to the Training Set
Algorithm.append('Support Vector Regression')
clf = SVR()
clf.fit(x_train, y_train)

# Predicting the val2 Results
predicted = clf.predict(x_val2)


# In[97]:


# Appending the Scores For Visualisation at a Later Part
MSE_Score.append(mean_squared_error(y_val2, predicted))
R2_Score.append(r2_score(y_val2, predicted))


# In[98]:


print("RMSE: ", math.sqrt(mean_squared_error(y_val2, predicted)))
print("Score: ", (clf.score(x_val2, y_val2))*100, "%")


# In[99]:


# Fitting Decision Tree to the Training Set
Algorithm.append('Decision Tree Regressor')
clf = DecisionTreeRegressor()
clf.fit(x_train, y_train)

# Predicting the val2 Results
predicted = clf.predict(x_val2)


# In[100]:


MSE_Score.append(mean_squared_error(y_val2, predicted))
R2_Score.append(r2_score(y_val2, predicted))


# In[101]:


print("RMSE: ", math.sqrt(mean_squared_error(y_val2, predicted)))
print("Score: ", (clf.score(x_val2, y_val2))*100, "%")


# In[102]:


# Fitting Random Forest Regressor Tree to the Training Set
Algorithm.append('Random Forest Regressor')
clf = RandomForestRegressor()
clf.fit(x_train, y_train)

# Predicting the val2 Results
predicted = clf.predict(x_val2)


# In[103]:


# Appending the Scores For Visualisation at a Later Part
MSE_Score.append(mean_squared_error(y_val2, predicted))
R2_Score.append(r2_score(y_val2, predicted))


# In[104]:


print("RMSE: ", math.sqrt(mean_squared_error(y_val2, predicted)))
print("Score: ", (clf.score(x_val2, y_val2))*100, "%")


# In[105]:


# Fitting XGBoost Regressor to the Training Set
Algorithm.append('XGB Regressor')
clf = xgb.XGBRegressor()
clf.fit(x_train, y_train)

# Predicting the Validation2 Results
predicted = clf.predict(x_val2)


# In[106]:


# Appending the Scores For Visualisation at a Later Part
MSE_Score.append(mean_squared_error(y_val2, predicted))
R2_Score.append(r2_score(y_val2, predicted))


# In[107]:


print("RMSE: ", math.sqrt(mean_squared_error(y_val2, predicted)))
print("Score: ", (clf.score(x_val2, y_val2))*100, "%")


# In[108]:


# Just Combining the Lists into a DataFrame for a Better Visualisation
Comparison = pd.DataFrame(list(zip(Algorithm, MSE_Score, R2_Score)), columns = ['Algorithm', 'MSE_Score', 'R2_Score'])


# In[109]:


#Score and Error
Comparison


# # Feeding test data into the trained model and getting the predicted delay.

# In[110]:


# Fitting SVR to the Test Set
clf = SVR()
clf.fit(x_train, y_train)

# Predicting the Test Set Results
predicted = clf.predict(x_test)


# In[111]:


#Adding the predicted delay colunm to the test_final data set.
test_final['delay'] = predicted


# In[112]:


len(predicted)


# In[113]:


test.head()


# In[114]:


test_final.shape


# In[121]:


test_final.drop('clear_date',axis = 1, inplace = True)


# In[122]:


#Calculating predicted clear date.
import datetime as dt
test_final['predicted_clear_date'] = test_final['due_in_date'] + test_final['delay'].map(dt.timedelta)


# In[129]:


#Calculating aging bucket.
test_final['Aging_Bucket'] = ""
test_final.loc[test_final['delay'].apply(int)>60,'Aging_Bucket'] = ">60"
test_final.loc[(test_final['delay'].apply(int)>= 46) & (test_final['delay'].apply(int)<= 60),'Aging_Bucket'] ="46-60"
test_final.loc[(test_final['delay'].apply(int)>= 31) & (test_final['delay'].apply(int)<= 45),'Aging_Bucket'] ="31-45"
test_final.loc[(test_final['delay'].apply(int)<= 30) & (test_final['delay'].apply(int)>= 16),'Aging_Bucket'] ="16-30"
test_final.loc[(test_final['delay'].apply(int)<= 15) & (test_final['delay'].apply(int)>= 0),'Aging_Bucket'] = "0-15"
test_final.loc[test_final['delay'].apply(int)< 0,'Aging_Bucket'] = "No Delay"


# In[131]:


test_final.groupby('Aging_Bucket').count()


# # Final Test Set with predicted clear date and bucketization column.

# In[130]:


test_final.head()


# In[ ]:




