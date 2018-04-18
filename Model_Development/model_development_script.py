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

def functions_ignitor(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY):
    print('MAIN FUNCTION TRIGGERED')
    download_filePath = downloadFromS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    print('DOWNLOADED DATA DUMP FROM S3')
    X_train, X_test, y_train, y_test = data_transformation(download_filePath)
    print('DATA TRANSFORMATION COMPLETED')
    upload_filePath = train_pickle_model(X_train, X_test, y_train, y_test)
    print('TRAINED & PICKLED MODEL')
    uploadToS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, upload_filePath)
    print('UPLOADED PICKLE TO S3')


def downloadFromS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY):
    bucket_name = 'ads-final-project-data-dump'
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    bucket = conn.get_bucket(bucket_name)
    bucket_list = bucket.list()
    for l in bucket_list:
        print('l: ', l)
        keyString = str(l.key)
        print('keyString: ', keyString)
        download_filePath = 'trainDataFromS3/'+keyString
        l.get_contents_to_filename('trainDataFromS3/'+keyString)

    return download_filePath

def uploadToS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, upload_filePath, destinationPath = ''):
    bucket_name = 'ads-final-project'
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
    bucket = conn.create_bucket(bucket_name,location=boto.s3.connection.Location.DEFAULT)
    print ('Uploading '+upload_filePath+' to Amazon S3 bucket '+bucket_name)
    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()   
    k = Key(bucket)
    k.key = destinationPath+"/"+upload_filePath
    k.set_contents_from_filename(upload_filePath, cb = percent_cb, num_cb = 10)
    print('Uploaded')

def data_transformation(download_filePath):
    df = pd.read_csv(download_filePath)
    #Replacing empty spaces with Null values
    df = df.replace(r'^\s+$', np.nan, regex=True)
    # Dropping NA values
    df = df.dropna()
    # Change the 'SeniorCitizen' variable from interger to categorial
    df['SeniorCitizen']=pd.Categorical(df['SeniorCitizen'])
    # Change the 'TotalCharges' variable from object to interger 
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'])
    # Deleting the custumerID column
    del df["customerID"]
    #Splitting data according to datatypes
    num = ['float64', 'int64']
    num_df = df.select_dtypes(include=num)
    obj_df = df.select_dtypes(exclude=num)
    # Add the 'Churn' variable in numeric dataset
    num_df = pd.concat([num_df,df["Churn"]],axis=1)
    #Creating bins and plotting Countplot for 'tenure'
    tenure_bins=pd.cut(num_df["tenure"], bins=[0,20,60,80], labels=['low','medium','high'])
    #Creating bins and plotting Countplot for 'MonthlyCharges'
    MonthlyCharges_bins=pd.cut(num_df["MonthlyCharges"], bins=[0,35,60,130], labels=['low','medium','high'])
    #Creating bins and plotting Countplot for 'MonthlyCharges'
    TotalCharges_bins=pd.cut(num_df["TotalCharges"], bins=[0,1000,4000,10000], labels=['low','medium','high'])
    #Saving bins into dataframe
    bins=pd.DataFrame([tenure_bins, MonthlyCharges_bins, TotalCharges_bins]).T
    #Converting SeniorCitizen variable into categorical and mapping values of 1 & 0 to Yes & No respectively
    df['SeniorCitizen'] = df.SeniorCitizen.map({0:'No', 1:'Yes'})
    # Concatenate bins with object variables
    df=pd.concat([bins,obj_df],axis=1)
    # Convert all the variables into categorical
    for i in list(df.columns):
        df[i] = pd.Categorical(df[i]) 
    dummy = pd.get_dummies(df) # Transform the categorical variables into dummy variables
    # Split training and testing dataset
    features = dummy.drop(["Churn_Yes", "Churn_No"], axis=1).columns
    X = dummy[features]
    y = dummy["Churn_Yes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

    return X_train, X_test, y_train, y_test

def train_pickle_model(X_train, X_test, y_train, y_test):
    # Training Logistic Regression Mode
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    predictions = logistic_regression.predict(X_test)
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    upload_filePath = 'logistic_regression.pkl'

    with open(upload_filePath, "wb") as fp:
        pickle.dump(logistic_regression, fp, protocol=2)

    return upload_filePath


if __name__ == '__main__':
    AWS_ACCESS_KEY_ID = sys.argv[1]
    AWS_SECRET_ACCESS_KEY = sys.argv[2]
    functions_ignitor(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    print('PROGRAM EXECUTED SUCCESSFULLY')