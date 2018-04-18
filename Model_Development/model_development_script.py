
# coding: utf-8

# In[174]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
import boto
from boto.s3.key import Key
from boto.s3.connection import Location
import os, sys, time


# In[144]:


# Loading the data
# df = pd.read_excel('Customer-Churn-Dataset.xls')
# X_train, X_test = train_test_split(df, test_size=0.20, random_state=7)
# X_train.to_csv('churn_train.csv', index=False)
# X_test.to_csv('churn_test.csv', index=False)

# Initial Split of Data
df = pd.read_csv('churn_train.csv')


# In[145]:


#Replacing empty spaces with Null values
df = df.replace(r'^\s+$', np.nan, regex=True)


# In[146]:


# Dropping NA values
df = df.dropna()


# In[147]:


# Change the 'SeniorCitizen' variable from interger to categorial
df['SeniorCitizen']=pd.Categorical(df['SeniorCitizen'])

# Change the 'TotalCharges' variable from object to interger 
df['TotalCharges']=pd.to_numeric(df['TotalCharges'])


# In[148]:


# Deleting the custumerID column
del df["customerID"]


# In[150]:


#Splitting data according to datatypes
num = ['float64', 'int64']
num_df = df.select_dtypes(include=num)
obj_df = df.select_dtypes(exclude=num)


# In[151]:


# Add the 'Churn' variable in numeric dataset
num_df = pd.concat([num_df,df["Churn"]],axis=1)


# In[152]:


#Creating bins and plotting Countplot for 'tenure'
tenure_bins=pd.cut(num_df["tenure"], bins=[0,20,60,80], labels=['low','medium','high'])


# In[153]:


#Creating bins and plotting Countplot for 'MonthlyCharges'
MonthlyCharges_bins=pd.cut(num_df["MonthlyCharges"], bins=[0,35,60,130], labels=['low','medium','high'])


# In[154]:


#Creating bins and plotting Countplot for 'MonthlyCharges'
TotalCharges_bins=pd.cut(num_df["TotalCharges"], bins=[0,1000,4000,10000], labels=['low','medium','high'])


# In[155]:


#Saving bins into dataframe
bins=pd.DataFrame([tenure_bins, MonthlyCharges_bins, TotalCharges_bins]).T


# In[156]:


#Converting SeniorCitizen variable into categorical and mapping values of 1 & 0 to Yes & No respectively
df['SeniorCitizen'] = df.SeniorCitizen.map({0:'No', 1:'Yes'})


# In[157]:


# Concatenate bins with object variables
df=pd.concat([bins,obj_df],axis=1)

# Convert all the variables into categorical
for i in list(df.columns):
    df[i] = pd.Categorical(df[i]) 
dummy = pd.get_dummies(df) # Transform the categorical variables into dummy variables


# In[158]:


# Split training and testing dataset
features = dummy.drop(["Churn_Yes", "Churn_No"], axis=1).columns


# In[159]:


X = dummy[features]
y = dummy["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)


# In[162]:


# Training Logistic Regression Model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
predictions = logistic_regression.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[163]:


with open('logistic_regression.pkl', "wb") as fp:
    pickle.dump(logistic_regression, fp, protocol=2)


# In[164]:


AWS_ACCESS_KEY_ID = 'AKIAJGQVKHVCVGELJALA'
AWS_SECRET_ACCESS_KEY = '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'


# In[175]:


def uploadToS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, filePath, destinationPath = ''):
    
    bucket_name = 'ads-final-project' + time.strftime("%y%m%d%H%M%S") + '-dump'
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
    bucket = conn.create_bucket(bucket_name,location=boto.s3.connection.Location.DEFAULT)
    
    print ('Uploading '+filePath+' to Amazon S3 bucket '+bucket_name)
    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()
        
    k = Key(bucket)
    k.key = destinationPath+"/"+filePath
    k.set_contents_from_filename(filePath, cb = percent_cb, num_cb = 10)
    print('Uploaded')


# In[177]:


uploadToS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, 'logistic_regression.pkl')

