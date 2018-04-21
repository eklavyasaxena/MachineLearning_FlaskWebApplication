import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from boruta import BorutaPy
import pickle
import boto
from boto.s3.key import Key
from boto.s3.connection import Location
import os, sys, time

DATA_DIRECTORY = 'trainingDataFromS3'
UPLOAD_DIRECTORY = 'uploadDataToS3'
BUCKET_NAME = 'ads-final-project'

def functions_ignitor(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY):
    create_directory(DATA_DIRECTORY)
    create_directory(UPLOAD_DIRECTORY)
    print('MAIN FUNCTION TRIGGERED')
    
    downloadFromS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    print('DOWNLOADED DATA DUMP FROM S3')
    
    X_all, y_all, df_dummies, upload_indexFilePath = data_transformation()
    print('DATA TRANSFORMATION COMPLETED')
    
    X_selected, y_selected, upload_featuredIndexFilePath = feature_engineering(X_all, y_all, df_dummies)
    print('FEATURE ENGINEERING COMPLETED')
    
    trained_models_with_rank, upload_metricsFilePath = model_training(X_selected, y_selected)
    print('MODEL TRAINING COMPLETED')    
    
    upload_modelFilePath = pickle_trained_model(trained_models_with_rank)
    print('PICKLED ALL MODELS')    
    
    uploadToS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, upload_indexFilePath)
    print('UPLOADED PICKLE INDEX TO S3')
     
    uploadToS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, upload_featuredIndexFilePath)
    print('UPLOADED PICKLE FEATURED INDEX TO S3')
    
    uploadToS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, upload_metricsFilePath)
    print('UPLOADED METRICS CSV TO S3')
    
    uploadToS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, upload_modelFilePath)
    print('UPLOADED PICKLE MODEL TO S3')
    
def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

def downloadFromS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY):
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    bucket = conn.get_bucket(BUCKET_NAME)
    bucket_list = bucket.list()
    for l in bucket_list:
        print('l: ', l)
        keyString = str(l.key)
        print('keyString: ', keyString)
        l.get_contents_to_filename(keyString)

def uploadToS3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, upload_filePath, destinationPath = ''):
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
    bucket = conn.create_bucket(BUCKET_NAME,location=boto.s3.connection.Location.DEFAULT)
    print ('Uploading '+upload_filePath+' to Amazon S3 bucket '+BUCKET_NAME)
    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()   
    k = Key(bucket)
    k.key = destinationPath+"/"+upload_filePath
    k.set_contents_from_filename(upload_filePath, cb = percent_cb, num_cb = 10)
    print('Uploaded')

def data_transformation():
    df = pd.read_csv(DATA_DIRECTORY+'/churn_train.csv')
    # Pickle all_features for Form Uploads
    df_to_index_pickle = df.drop(["Churn"], axis=1)
    upload_indexFilePath = pickle_df_index(df_to_index_pickle, 'index_dict.pkl')
    # Replacing empty spaces with Null values
    df = df.replace(r'^\s+$', np.nan, regex=True)
    # Dropping NA values
    df = df.dropna()
    # Deleting the custumerID column
    del df["customerID"]
    # Removing TotalCharges variable from the data
    del df["TotalCharges"]
    #Converting SeniorCitizen variable into categorical and mapping values of Yes & No to 1 & 0 respectively
    df['SeniorCitizen'] = df.SeniorCitizen.map({0:'No', 1:'Yes'})
    all_columns_list = df.columns.tolist()
    numerical_columns_list = ['tenure','MonthlyCharges']
    categorical_columns_list = [e for e in all_columns_list if e not in numerical_columns_list]
    for index in categorical_columns_list:
        df[index] = pd.Categorical(df[index])
    for index in numerical_columns_list:
        df[index] = pd.to_numeric(df[index])
    # Splitting data according to datatypes
    num = ['float64', 'int64']
    num_df = df.select_dtypes(include=num)
    obj_df = df.select_dtypes(exclude=num)
    # Creating bins for numerical variables for extensive prediction of churn
    tenure_bins = pd.cut(num_df["tenure"], bins=[0,20,60,80], labels=['low','medium','high'])
    MonthlyCharges_bins = pd.cut(num_df["MonthlyCharges"], bins=[0,35,60,130], labels=['low','medium','high'])
    # Saving numeric variable bins into a dataframe
    bins = pd.DataFrame([tenure_bins, MonthlyCharges_bins]).T
    # Concatenate bins with object variables
    transformed_df = pd.concat([bins,obj_df],axis=1)
    dummy_columns = [e for e in transformed_df.columns if e != 'Churn']
    # Creating dataframe of dummy variables
    df_dummies = pd.get_dummies(data=transformed_df, columns=dummy_columns)
    df_dummies_features = df_dummies.drop(["Churn"], axis=1).columns
    X_all = df_dummies[df_dummies_features]
    y_all = df_dummies["Churn"]
    return X_all, y_all, df_dummies, upload_indexFilePath

def feature_engineering(X_all, y_all, df_dummies):
    # Change X and y to its values
    X_boruta = X_all.values
    y_boruta = y_all.values
    # Define random forest classifier, with utilising all cores and sampling in proportion to y labels
    rfc = RandomForestClassifier(n_jobs = -1)
    # Define Boruta feature selection method
    feature_selector = BorutaPy(rfc, n_estimators='auto', random_state=1)
    # Find all relevant features
    feature_selector.fit(X_boruta, y_boruta)
    #Transposing dataframe for ranking
    df_features_rank = df_dummies.drop(['Churn'],axis=1).T
    # Check ranking of features
    df_features_rank['Boruta_Rank'] = feature_selector.ranking_
    # Adding a variable 'Feature' in the dataframe
    df_features_rank['Feature']=  df_features_rank.index
    # Sort the dataframe as per Rank
    df_features_rank = df_features_rank.sort_values('Boruta_Rank')
    # Exctracting only top 2 ranked features
    df_top2_ranked_feature = df_features_rank.loc[df_features_rank['Boruta_Rank'].isin([1,2])]
    # Selecting important featutres
    selected_features = df_top2_ranked_feature.index
    X_selected = df_dummies[selected_features]
    y_selected = df_dummies["Churn"]
    # Pickle the selected features for Form Uploads
    upload_featuredIndexFilePath = pickle_df_index(X_selected, 'featured_index_dict.pkl')    
    return X_selected, y_selected, upload_featuredIndexFilePath

def model_training(X_selected, y_selected):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_selected, y_selected, test_size=0.20, random_state=7)
    # Make predictions on test dataset
    models = []
    accuracy_list = []
    trained_models = {}
    models.append(('LogisticRegression', LogisticRegression()))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    models.append(('RandomForestClassifier', RandomForestClassifier()))
    for name, model in models:
        model.fit(X_train, y_train)
        trained_models[name] = model
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        accuracy_list.append((name,acc))
    #Creating a dataframe for the models metrics
    models_metrics = pd.DataFrame(accuracy_list, columns=["Model", "Accuracy"]) 
    models_metrics['Model_Rank'] = models_metrics['Accuracy'].rank(ascending=False, method='first')
    # Store the result into csv
    upload_metricsFilePath = UPLOAD_DIRECTORY+'/metrics_score.csv'
    models_metrics.to_csv(upload_metricsFilePath, index=False)
    # Compiling all the models in single dictionary
    rank_dict = pd.Series(models_metrics.Model_Rank.values, index=models_metrics.Model.values).to_dict()
    trained_models_with_rank = {}
    for key, value in rank_dict.items():
        trained_models_with_rank[rank_dict[key]] = [value1 for key1, value1 in trained_models.items() if key == key1]
        trained_models_with_rank[rank_dict[key]].append(key)
    return trained_models_with_rank, upload_metricsFilePath

def pickle_trained_model(trained_models_with_rank):
    upload_modelFilePath = UPLOAD_DIRECTORY+'/pickled_models.pkl'
    with open(upload_modelFilePath, "wb") as fp:
        pickle.dump(trained_models_with_rank, fp, protocol=2)
    return upload_modelFilePath

def pickle_df_index(df, filename):
    # Pickle index for Form Uploads
    index_dict = dict(zip(df.columns,range(df.shape[1])))
    upload_indexFilePath = UPLOAD_DIRECTORY+'/'+filename
    with open(upload_indexFilePath, "wb") as fp:
        pickle.dump(index_dict, fp, protocol=2)
    return upload_indexFilePath

if __name__ == '__main__':
    AWS_ACCESS_KEY_ID = sys.argv[1]
    AWS_SECRET_ACCESS_KEY = sys.argv[2]
    functions_ignitor(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    print('PROGRAM EXECUTED SUCCESSFULLY')