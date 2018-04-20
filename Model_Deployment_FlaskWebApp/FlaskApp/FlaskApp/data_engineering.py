import pandas as pd
import numpy as np
import boto3
import pickle
from dask.distributed import Client
from werkzeug.utils import secure_filename
import time
import os

BUCKET_NAME = 'ads-final-project'
PICKLED_MODELS = ['logistic_regression.pkl']

try:
	S3 = boto3.client('s3', region_name='us-east-1')
except Exception as e:
	print(str(e))
	raise e

def fetch_accuracy_metrics_FromS3(key):
	try:
		response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
		accuracy_metrics = response['Body'].read()
		accuracy_metrics_df = pd.read_csv(accuracy_metrics)
		return accuracy_metrics_df
	except Exception as e:
		print(str(e))
		raise e

def fetch_pickle_FromS3(key):
	try:
		response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
		pickled_body = response['Body'].read()
		result = pickle.loads(pickled_body)
		return result
	except Exception as e:
		print(str(e))
		raise e

def data_transformation(df, featured_index_dict):
	try:
		df = df.replace(r'^\s+$', np.nan, regex=True)
		df = df.dropna()
		try:
			del df["Churn"]
		except:
			pass
		del df["customerID"]
		all_columns_list = df.columns.tolist()
		numerical_columns_list = ['tenure', 'TotalCharges','MonthlyCharges']
		categorical_columns_list = [e for e in all_columns_list if e not in numerical_columns_list]
		for index in categorical_columns_list:
			df[index] = pd.Categorical(df[index])
		for index in numerical_columns_list:
			df[index] = pd.to_numeric(df[index])
		num = ['float64', 'int64']
		num_df = df.select_dtypes(include=num)
		obj_df = df.select_dtypes(exclude=num)
		tenure_bins=pd.cut(num_df["tenure"], bins=[0,20,60,80], labels=['low','medium','high'])
		MonthlyCharges_bins=pd.cut(num_df["MonthlyCharges"], bins=[0,35,60,130], labels=['low','medium','high'])
		TotalCharges_bins=pd.cut(num_df["TotalCharges"], bins=[0,1000,4000,10000], labels=['low','medium','high'])
		bins=pd.DataFrame([tenure_bins, MonthlyCharges_bins, TotalCharges_bins]).T
		# df['SeniorCitizen'] = df.SeniorCitizen.map({0:'No', 1:'Yes'})
		df=pd.concat([bins,obj_df],axis=1)
		for i in list(df.columns):
		    df[i] = pd.Categorical(df[i]) 
		dummy = pd.get_dummies(df)
		features_not_in_dummy = [key for key in featured_index_dict if key not in list(dummy.columns)]
		for l in features_not_in_dummy:
			dummy[l] = 0
		'''For Feature and Dummy Index Comparisons
		dummy_columns = sorted(list(dummy.columns))
		features_columns = sorted([key for key in featured_index_dict])
		diff = list(set(dummy_columns) - set(features_columns))
		print('dummy: ', dummy_columns)
		print('----------------------------')
		print('features: ', features_columns)
		print('----------------------------')
		print('diff: ', diff)
		'''
		features = dummy.columns
		print(features)
		data_X = dummy[features]
		print('0) data_X.shape: ', data_X.shape)
		return data_X
	except Exception as e:
		print(e)

def data_processing(input_file):
	try:
		featured_index_dict = fetch_pickle_FromS3('featured_index_dict.pkl')
		data_dataframe = pd.read_csv(input_file)
		total_rows = data_dataframe.shape[0]
		data_X = data_transformation(data_dataframe, featured_index_dict)
		print('1) data_X.shape: ', data_X.shape)
		for i in range(0, len(PICKLED_MODELS)):
			# Load Model
			model = fetch_pickle_FromS3(PICKLED_MODELS[i])
			model_name = secure_filename(PICKLED_MODELS[i]).rsplit('.', 1)[0]
			# Make prediction
			if total_rows > 5:
				client = Client(processes=False)
				print('2-if) data_X.shape: ', data_X.shape)
				prediction = client.submit(model.predict, data_X).result().tolist()
			else:
				print('2-else) data_X.shape: ', data_X.shape)
				prediction = model.predict(data_X).tolist()
			print('3) data_X.shape: ', data_X.shape)
			prediction_series = pd.Series(prediction)
			data_dataframe[model_name] = prediction_series
		return data_dataframe
	except Exception as e:
		print(str(e))
		raise e

def df_to_html(df):
	df_html = df.to_html(index=False)
	df_html = df_html.replace('<table border="1" class="dataframe">','<table class="table">')
	df_html = df_html.replace('<thead>','<thead class="thead-dark">')
	df_html = df_html.replace('<tr style="text-align: right;">','<tr>')
	df_html = df_html.replace('<th>','<th scope="col">')
	return df_html

def form_processing(input_dict):
	try:
		form_result_df = pd.DataFrame.from_dict(input_dict)
		small_df = form_result_df.filter(['customerID'], axis=1)
		featured_index_dict = fetch_pickle_FromS3('featured_index_dict.pkl')
		data_X = data_transformation(form_result_df, featured_index_dict)
		for i in range(0, len(PICKLED_MODELS)):
			# Load Model
			model = fetch_pickle_FromS3(PICKLED_MODELS[i])
			model_name = secure_filename(PICKLED_MODELS[i]).rsplit('.', 1)[0]
			# Make prediction
			print('1) data_X.shape: ', data_X.shape)
			prediction = model.predict(data_X).tolist()
			print('3) data_X.shape: ', data_X.shape)
			prediction_series = pd.Series(prediction)
			form_result_df[model_name] = prediction_series
			small_df[model_name] = prediction_series
		return form_result_df, small_df
		'''May useful for JSON API
		index_dict = fetch_pickle_FromS3('index_dict.pkl')
		data_dict = {}
		for key, value in index_dict.items():
			data_dict[key] = input_dict[key][0]
		'''
	except Exception as e:
		print(str(e))
		raise e