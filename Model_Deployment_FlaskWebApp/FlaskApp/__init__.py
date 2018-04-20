from flask import Flask, render_template, flash, request, url_for, \
	redirect, session, make_response, send_file, send_from_directory, \
	jsonify, abort, Response
from wtforms import Form, BooleanField, TextField, PasswordField, validators
from passlib.hash import sha256_crypt
from flaskext.mysql import MySQL
from functools import wraps
import gc
import smtplib
from flask_mail import Mail, Message
import os
from werkzeug.utils import secure_filename
import sys
import time
import datetime
from dask.distributed import Client, LocalCluster, Scheduler
import pygal
import pickle
import numpy as np
import pandas as pd
# from dbconnect import connection
# from MySQLdb import escape_string as thwart

# from flask_debugtoolbar import DebugToolbarExtension
# toolbar = DebugToolbarExtension(app)

app = Flask(__name__, instance_path = '/var/www/FlaskApp/FlaskApp/protected')
app.secret_key = 'eiuweui2__478rw[[ioj4<6h09krv#%#Y$$^Gr'

app.config.update(
	DEBUG=True,
	#Email Settings
	MAIL_SERVER = 'smtp.gmail.com',
	MAIL_PORT = 465,
	MAIL_USE_SSL = True,
	MAIL_USERNAME = 'your@gmail.com',
	MAIL_PASSWORD = 'yourpassword'
	)
mail = Mail(app)

mysql = MySQL()
# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'machineflask'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
# app.config['MYSQL_DATABASE_PASSWORD'] = ''
mysql.init_app(app)

conn = mysql.connect()
c = conn.cursor()

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_FILE_SIZE'] = 10000000 #1MB limit

if False:
	import boto
	import sys, os
	from boto.s3.key import Key

	AWS_ACCESS_KEY_ID = ''
	AWS_SECRET_ACCESS_KEY = ''

	bucket_name = 'adsmodeldevelopmentdeployment'
	# connect to the bucket
	conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
	bucket = conn.get_bucket(bucket_name)
	# go through the list of files
	bucket_list = bucket.list()
	for l in bucket_list:
	  keyString = str(l.key)
	  l.get_contents_to_filename(UPLOAD_FOLDER+keyString)

def login_required(f):
	@wraps(f)
	def wrap(*args, **kwargs):
		if 'logged_in' in session:
			return f(*args, **kwargs)
		else:
			flash('Login Required!!!')
			return redirect(url_for('batchfilelogin'))
	return wrap

def special_requirement(f):
	@wraps(f)
	def wrap(*args, **kwargs):
		try:
			if 1 == session['rank']:
				return f(*args, **kwargs)
			else:
				flash('Please REGISTER and SUBSCRIBE to access Batch File Services')
				return redirect(url_for('ontheflyform'))
		except:
			flash('Please REGISTER and SUBSCRIBE to access Batch File Services')
			return redirect(url_for('ontheflyform'))
	return wrap

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class RegistrationForm(Form):
	#username = TextField('Username', [validators.Length(min=4, max=20)])
	email = TextField('Email Address', [validators.Length(min=6, max=50)])
	password = PasswordField('Password', [validators.Required(), validators.Length(min=6, max=50),
		validators.EqualTo('confirm', message='Passwords Must Match!')])
	confirm = PasswordField('Confirm Password')
	accept_tos = BooleanField('I acknowledge that subscription fee has to be paid post registration to access the Batch File Services.', [validators.Required()])


@app.route('/')
def homepage():
	print('homepage route executed')
	return render_template('main.html')

@app.route('/logout/')
@login_required
def logout():
	session.clear()
	flash('Succesfully Logged Out!!!')
	gc.collect()
	return redirect(url_for('batchfilelogin'))

@app.route('/batchfilelogin/', methods=['GET', 'POST'])
def batchfilelogin():
	error = ''
	try:
		# c, conn = connection()
		if request.method == 'POST':
			print('Email: ', request.form['email'])
			c.execute('''SELECT * FROM users WHERE email = %s''', (request.form['email'], ))

			data = c.fetchone()
			print('Data Fetched: ', data)

			password = data[2]
			rank = data[3]

			if sha256_crypt.verify(request.form['password'], password):
				session['logged_in'] = True
				session['email'] = request.form['email']
				session['rank'] = rank

				flash('You are now Logged In')
				return redirect(url_for('ontheflyform'))

			else:
				error = 'Invalid Credentials, Try Again!'

		gc.collect()

		return render_template('batchfilelogin.html', error=error)

	except Exception as e:
		print(e)
		error = 'Invalid Credentials, Try Again!'
		return render_template('batchfilelogin.html', error=error)

@app.route('/register/', methods=['GET', 'POST'])
def register_page():
	try:
		form = RegistrationForm(request.form)

		if request.method == 'POST' and form.validate():
			email = form.email.data
			password = sha256_crypt.encrypt((str(form.password.data)))
			# c, conn = connection()

			x = c.execute('''SELECT * FROM users WHERE email = %s''', (email, ))

			print('Printing x: ', x)

			if int(x) > 0:
				flash('This EmailID is already registered.')
				return render_template('register.html', form = form)
			else:
				c.execute('''INSERT INTO users (email, password) VALUES (%s, %s)''', (email, password))

				conn.commit()
				flash('Succesfully Registered. We will contact you soon with the subscription fees & charges.')
				c.close()
				conn.close()
				gc.collect()

				session['logged_in'] = True
				session['email'] = email

				return redirect(url_for('ontheflyform'))

		return render_template('register.html', form = form)

	except Exception as e:
		return(str(e))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

@app.route('/send-mail/')
def send_mail():
    try:
    	msg = Message('Send Mail Tester',
    		sender='yoursendingemail@gmail.com',
    		recipients=['recievingemail@email.com'])
    	msg.body = 'This is the Tester Mail'
    	mail.send(msg)
    	return 'Mail Sent!!!'

    except Exception as e:
    	return str(e)

@app.route('/secret/<path:filename>')
@special_requirement
def protected(filename):
	try:
		return send_from_directory(os.path.join(app.instance_path,''), filename)

	except Exception as e:
		return redirect(url_for('homepage'))

@app.route('/return-file/')
def return_file():
	return send_file('/var/www/FlaskApp/FlaskApp/data/header.csv',as_attachment=True, attachment_filename = 'inputformat.csv') 

@app.route('/interactive/')
def interactive():
	try:
		return render_template('interactive.html')
	except Exception as e:
		return (str(e))

@app.route('/background_process')
def background_process():
	try:
		lang = request.args.get('proglang', 0, type=str)
		if lang.lower() == 'python':
			return jsonify(result='You are wise')
		else:
			return jsonify(result='Try again.')
	except Exception as e:
		return str(e)

@app.route('/pygalexample/')
def pygalexample():
	try:
		graph = pygal.Line()
		graph.title = '% Change Coolness of programming languages over time.'
		graph.x_labels = ['2011','2012','2013','2014','2015','2016']
		graph.add('Python',  [15, 31, 89, 200, 356, 900])
		graph.add('Java',    [15, 45, 76, 80,  91,  95])
		graph.add('C++',     [5,  51, 54, 102, 150, 201])
		graph.add('All others combined!',  [5, 15, 21, 55, 92, 105])
		graph_data = graph.render_data_uri()
		return render_template('graphing.html', graph_data=graph_data)
	except Exception as e:
		return (str(e))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def data_transformation(df):
    cols = [68,20,44,56,59,6,45,13,37,54,22,16,10,80,12,2,23,77,63,36,26,35,75,33,14,30,27,38,52,57,65,66,43,32,1,46,55,7,49,34,60,67,9,73,21,53,78,79,70,50,64,4,61,62,58,74,48,25,76,71,41,0,18,69,5,31,40,28,8,19,47,51,29,3,11,72,17,24,39,15,42]
    
    df = df.rename(columns = {'class' : 'Flag'})
    df['Flag'] = df.Flag.map({'neg':0, 'pos':1})
    df = df.replace(['na'],np.nan)
    
    df.to_csv(UPLOAD_FOLDER + '/step1.csv', index=False)
    # Principal Component Analysis
    df_X = df.loc[:,df.columns != 'Flag']
    df_Y = df.loc[:,df.columns == 'Flag']
    
    df_X = df_X.apply(pd.to_numeric)
    df_X= df_X.fillna(df_X.median()).dropna(axis =1 , how ='all')

    df_X.to_csv(UPLOAD_FOLDER + '/step1.csv', index=False)
    
    scaler = StandardScaler()
    scaler.fit(df_X)
    df_X = scaler.transform(df_X)
    df_X = pd.DataFrame(df_X)

    df_X.to_csv(UPLOAD_FOLDER + '/step3.csv', index=False)
    df_X = df_X[cols]
    
    return df_X

@app.route('/batchfileupload/')
@special_requirement
def batchfile_home():
	try:
		flash('Please choose a .csv File and hit UPLOAD')
		return render_template('batchfileupload.html')
	except Exception as e:
		return(str(e))

@app.route('/getprediction/', methods=['GET', 'POST'])
@special_requirement
def get_prediction():
	responses = ''
	tables = ''
	error = ''
	predictions_html = ''
	table = {}
	try:
		if request.method == 'POST':
			# check if the post request has the file part
			if 'file' not in request.files:
				flash('No File Selected')
				return render_template('batchfileupload.html')

			file = request.files.get('file')

			# if user does not select file, browser also submit a empty part without filename
			if file.filename == '':
				flash('No File Selected')
				return render_template('batchfileupload.html')

			if file and allowed_file(file.filename):
				'''Function to UPLOAD a File (not required)
				filename = secure_filename(file.filename)
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				os.rename(UPLOAD_FOLDER + filename, UPLOAD_FOLDER + 'uploadedByClient.csv')
				#return redirect(url_for('uploaded_file', filename=filename))
				flash('File Uploaded Succesfully')
				'''
				uploaded_df = pd.read_csv(file)
				test = data_transformation(uploaded_df)
				# test.to_csv('/var/tmp/final_check_model.csv', encoding='utf-8', index=False)

				try:
					loaded_model_1 = None
					with open(UPLOAD_FOLDER+'/Models/LR_model.pkl','rb') as f:
						loaded_model_1 = pickle.load(f)
					predictions_out_1 = loaded_model_1.predict(test)
					prediction_series_1 = pd.Series(predictions_out_1)

					# loaded_model_2 = None
					# with open('/var/tmp/Models/RandomForest.pkl','rb') as f:
					# 	loaded_model_2 = pickle.load(f)
					# predictions_out_2 = loaded_model_2.predict(test)
					# prediction_series_2 = pd.Series(predictions_out_2)

					# loaded_model_3 = None
					# with open('/var/tmp/Models/SupportVectorMachine.pkl','rb') as f:
					# 	loaded_model_3 = pickle.load(f)
					# predictions_out_3 = loaded_model_3.predict(test)
					# prediction_series_3 = pd.Series(predictions_out_3)


					test['LogisticRegression'] = prediction_series_1
					# test['RandomForest'] = prediction_series_2
					# test['SupportVectorMachine'] = prediction_series_3
					# test[loaded_model_name] = prediction_series

					predictions = test.to_json(orient="records")
					predictions_html = test.to_html(index=False)

					responses = jsonify(predictions=predictions)
					responses.status_code = 200
					table['header'] = test.columns.tolist()
					table['values'] = test.values

				except Exception as e:
					return(str(e))

			else:
				print('Invalid File Format, Try Again!')
				flash('Invalid File Format, Try Again!')
				error = 'Invalid File Format, Try Again!'
				return render_template('batchfileupload.html')

		gc.collect()

		print('render_template prediction_result.html')
		flash('Download the csv File')
		return render_template('prediction_result.html', responses=responses, tables=[predictions_html], error=error)

	except Exception as e:
		return(str(e))



@app.route('/ontheflyform/')
def ontheflyform():
	try:
		flash('Please complete the form and hit SUBMIT')
		return render_template('ontheflyform.html')
	except Exception as e:
		return(str(e))

@app.route('/getsingleprediction/',methods=['POST','GET'])
def get_single_prediction():
	prediction = ''
	try:
		if request.method=='POST':

			result = request.form
			single_input = str(result['single_input'])
			input_list = [single_input.split(",")]

			'''Use to extract header from the main file without predictive value
			df = pd.read_csv("energy_training.csv")
			df = df.drop(['<something>'],axis=1)
			df = df.head(0)
			df.to_csv('header.csv', encoding='utf-8', index=False)
			'''

			df_for_header = pd.read_csv(UPLOAD_FOLDER+"/data/header.csv")
			df_header = df_for_header.columns.tolist()

			df = pd.DataFrame(input_list, columns=df_header)
			df.to_csv(UPLOAD_FOLDER+'/in_process.csv', index=False)
			new_df = pd.read_csv(UPLOAD_FOLDER+'/in_process.csv')
			
			test = data_transformation(new_df)

			test.to_csv(UPLOAD_FOLDER+'/transf_in_process.csv', index=False)
			test = pd.read_csv(UPLOAD_FOLDER+'/transf_in_process.csv')
			

			try:
				loaded_model = None
				with open(UPLOAD_FOLDER+'/models/LR_model.pkl','rb') as f:
					loaded_model = pickle.load(f)

				predictions_out = loaded_model.predict(test)

				# prediction_series = pd.Series(predictions_out)
				# test['Predictions'] = prediction_series

				# # predictions = test.to_json(orient="records")
				# predictions_html = test.to_html(index=False)

			except Exception as e:
				return(str(e))
			
		gc.collect()
		# return(str(input_list))
		flash('Following are the predictions')
		return render_template('onfly_result.html', tables=predictions_out[0])

	except Exception as e:
		return (str(e))

if __name__ == '__main__':
    app.run()