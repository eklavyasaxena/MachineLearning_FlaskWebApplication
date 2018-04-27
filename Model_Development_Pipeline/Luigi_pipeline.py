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
from os import listdir
from os.path import isfile, join
import luigi

class data_transformation(luigi.Task):


    def run(self):
    	df = pd.read_csv("Customer-Churn-Dataset.csv")
    	df_to_index_pickle = df.drop(["Churn"],axis=1)
    	df = df.replace(r'^\s+$', np.nan, regex=True)
    	df = df.dropna()
    	del df["customerID"]
    	del df["TotalCharges"]
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
    	transformed_df = pd.concat([bins,obj_df],axis=1)
    	dummy_columns = [e for e in transformed_df.columns if e != 'Churn']
    	df_dummies = pd.get_dummies(data=transformed_df, columns=dummy_columns)
    	df_dummies_features = df_dummies.drop(["Churn"], axis=1).columns
    	X_all = df_dummies[df_dummies_features]
    	y_all = df_dummies["Churn"]
    	df_dummies.to_csv(self.output()['output1'].path,index = False)
    	X_all.to_csv(self.output()['output2'].path,index = False)
    	y_all.to_csv(self.output()['output3'].path,index = False)
		#return X_all, y_all, df_dummies, upload_indexFilePath
		
    def output(self):
        return {'output1' : luigi.LocalTarget("data_transformation.csv"),
        		'output2' : luigi.LocalTarget("data_transformationx.csv"),
        		'output3' : luigi.LocalTarget("data_transformationy.csv") }


class feature_engineering(luigi.Task):

	def requires(self):
		return {'input1': data_transformation(), 'input2': data_transformation(), 'input3': data_transformation()}
		#yield data_transformation()

	def run(self):
		print("Here : ")
		df_dummies=pd.read_csv(data_transformation().output()['output1'].path)
		X_all=pd.read_csv(data_transformation().output()['output2'].path)
		y_all=pd.read_csv(data_transformation().output()['output3'].path)
		
		X_boruta = X_all.values
		y_boruta = y_all.values
		y_boruta = np.insert(y_boruta, 7031, 'NO')
		
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
		print(self.output())
		X_selected.to_csv(self.output()['output1'].path,index = False)
		y_selected.to_csv(self.output()['output2'].path,index = False)
       	#return X_selected, y_selected, upload_featuredIndexFilePath

	def output(self):
		return {'output1' : luigi.LocalTarget("feature_engineeringx.csv"),
        		'output2' : luigi.LocalTarget("feature_engineeringy.csv") }

class model_training(luigi.Task):

	def requires(self):
		return {'input1': feature_engineering(), 'input2': feature_engineering()}

	def run(self):
	
		X_selected = pd.read_csv(feature_engineering().output()['output1'].path)
		y_selected = pd.read_csv(feature_engineering().output()['output2'].path)
		X_selected=X_selected.drop(X_selected.index[7031])

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

		rank_dict = pd.Series(models_metrics.Model_Rank.values, index=models_metrics.Model.values).to_dict()
		trained_models_with_rank = {}
		for key, value in rank_dict.items():
			trained_models_with_rank[rank_dict[key]] = [value1 for key1, value1 in trained_models.items() if key == key1]
			trained_models_with_rank[rank_dict[key]].append(key)
    	#return trained_models_with_rank, upload_metricsFilePath

	def output(self):
		return luigi.LocalTarget("model_training.csv")

if __name__ == '__main__':
    luigi.run(['model_training'])