import pandas as pd
import numpy as np
import boto3
import pickle
from dask.distributed import Client
from werkzeug.utils import secure_filename
import time
import os, io

DATA_DIRECTORY = 'trainingDataFromS3'
UPLOAD_DIRECTORY = 'uploadDataToS3'
BUCKET_NAME = 'ads-final-project'

try:
	S3 = boto3.client('s3', region_name='us-east-1')
except Exception as e:
	print(str(e))
	raise e

def fetch_pickle_FromS3(key):
	try:
		response = S3.get_object(Bucket=BUCKET_NAME, Key=UPLOAD_DIRECTORY+'/'+key)
		pickled_body = response['Body'].read()
		result = pickle.loads(pickled_body)
		return result
	except Exception as e:
		print(str(e))
		raise e

def fetch_metrics_score_FromS3(key):
	try:
		response = S3.get_object(Bucket=BUCKET_NAME, Key=UPLOAD_DIRECTORY+'/'+key)
		metrics_score = response['Body'].read()
		metrics_score_df = pd.read_csv(io.BytesIO(metrics_score))
		metrics_score_df = metrics_score_df.sort_values('Model_Rank')
		return metrics_score_df
	except Exception as e:
		print(str(e))
		raise e

def df_to_html(df, changeClass=True):
	df_html = df.to_html(index=False)
	if changeClass:
		df_html = df_html.replace('<table border="1" class="dataframe">','<table class="table">')
	else:
		df_html = df_html.replace('<table border="1" class="dataframe">','<table border="2" class="dataframe">')
	df_html = df_html.replace('<thead>','<thead class="thead-dark">')
	df_html = df_html.replace('<tr style="text-align: right;">','<tr align="center" valign="center">')
	df_html = df_html.replace('<th>','<th scope="col">')
	df_html = df_html.replace('<td>','<td align="center" valign="center">')
	return df_html

def data_transformation(df, featured_index_dict):
	try:
		try:
			del df["Churn"]
		except:
			pass
		df = df.drop(["customerID"], axis=1)
		df = df.drop(["TotalCharges"], axis=1)
		df = df.replace(r'^\s+$', np.nan, regex=True)
		df = df.dropna()
		df['SeniorCitizen'] = df.SeniorCitizen.map({0:'No', 1:'Yes'})
		all_columns_list = df.columns.tolist()
		numerical_columns_list = ['tenure','MonthlyCharges']
		categorical_columns_list = [e for e in all_columns_list if e not in numerical_columns_list]
		for index in categorical_columns_list:
			df[index] = pd.Categorical(df[index])
		for index in numerical_columns_list:
			df[index] = pd.to_numeric(df[index])
		num = ['float64', 'int64']
		num_df = df.select_dtypes(include=num)
		obj_df = df.select_dtypes(exclude=num)
		tenure_bins = pd.cut(num_df["tenure"], bins=[0,20,60,80], labels=['low','medium','high'])
		MonthlyCharges_bins = pd.cut(num_df["MonthlyCharges"], bins=[0,35,60,130], labels=['low','medium','high'])
		bins = pd.DataFrame([tenure_bins, MonthlyCharges_bins]).T
		transformed_df = pd.concat([bins, obj_df],axis=1)

		df_dummies = pd.get_dummies(data=transformed_df, columns=transformed_df.columns)

		features_columns = sorted([key for key in featured_index_dict])
		features_not_in_dummy = [e for e in features_columns if e not in list(df_dummies.columns)]
		for l in features_not_in_dummy:
			df_dummies[l] = 0
		
		'''For Feature and Dummy Index Comparisons
		dummy_columns = sorted(list(df_dummies.columns))
		features_columns = sorted([key for key in featured_index_dict])
		subset = set(features_columns).issubset(set(dummy_columns))
		print('df_dummies: ', dummy_columns)
		print('----------------------------')
		print('features: ', features_columns)
		print('----------------------------')
		print('subset: ', subset)
		'''

		data_X = df_dummies[features_columns]
		print('0) data_X.shape: ', data_X.shape)
		return data_X
	except Exception as e:
		print(e)

def predict_outcome(data_X, data_dataframe):
	pickled_models = fetch_pickle_FromS3('pickled_models.pkl')
	total_rows = data_dataframe.shape[0]
	small_df = data_dataframe.filter(['customerID'], axis=1)
	for i in range(1, (len(pickled_models)+1)):
		print('2) data_X.shape.in_predict_outcome: ', data_X.shape)
		# Load Model
		model_rank = i
		print('model_rank: ', i)
		model = pickled_models[i][0]
		model_name = 'Rank '+str(model_rank)+': '+pickled_models[i][1]
		print('model_name: ', model_name)
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
		small_df[model_name] = prediction_series
	return data_dataframe, small_df

def batchfile_processing(input_file):
	try:
		featured_index_dict = fetch_pickle_FromS3('featured_index_dict.pkl')
		data_dataframe = pd.read_csv(input_file)
		print('1) input_file_columns: ', data_dataframe.columns)
		data_X = data_transformation(data_dataframe, featured_index_dict)
		print('22) input_file_columns: ', data_dataframe.columns)
		print('1) data_X.shape: ', data_X.shape)
		data_dataframe, small_df = predict_outcome(data_X, data_dataframe)

		return data_dataframe
	except Exception as e:
		print(str(e))
		raise e

def form_processing(input_dict):
	try:
		form_result_df = pd.DataFrame.from_dict(input_dict)
		featured_index_dict = fetch_pickle_FromS3('featured_index_dict.pkl')
		data_X = data_transformation(form_result_df, featured_index_dict)

		form_result_df, small_df = predict_outcome(data_X, form_result_df)

		return form_result_df, small_df

	except Exception as e:
		print(str(e))
		raise e