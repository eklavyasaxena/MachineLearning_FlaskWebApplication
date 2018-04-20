from flask import Flask, render_template, flash, request, url_for, \
	redirect, session, make_response, send_file, send_from_directory, \
	jsonify, abort, Response
from wtforms import Form, BooleanField, TextField, PasswordField, validators
from passlib.hash import sha256_crypt
from flaskext.mysql import MySQL
from functools import wraps
import gc, os, sys, time, datetime, csv
import smtplib
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from dask.distributed import Client, LocalCluster, Scheduler
import pygal
import pickle
import numpy as np, pandas as pd
from dask.distributed import Client
from flask_restful import Resource, Api
from flask_httpauth import HTTPBasicAuth
import data_engineering
import json

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

auth = HTTPBasicAuth()

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_FILE_SIZE'] = 10000000 #1MB limit

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

@app.route('/return-file/')
def return_file():
	return send_file(UPLOAD_FOLDER+'/data/header.csv',as_attachment=True, attachment_filename = 'inputformat.csv')

@app.route('/batchfileupload/')
@special_requirement
def batchfile_home():
	try:
		# Creating HEADER .csv file for Upload Format
		index_dict = data_engineering.fetch_pickle_FromS3('index_dict.pkl')
		index_list = [key for key, value in index_dict.items()]
		with open(UPLOAD_FOLDER+'/data/header.csv', "w") as output:
			writer = csv.writer(output, lineterminator='\n')
			writer.writerow(index_list) 
		flash('Please choose a .csv File and hit UPLOAD')
		return render_template('batchfileupload.html')
	except Exception as e:
		return(str(e))

@app.route('/getprediction/', methods=['GET', 'POST'])
@special_requirement
def get_prediction():
	tables = ''
	error = ''
	try:
		if request.method == 'POST':
			# check if post has the file part
			if 'file' not in request.files:
				flash('No File Selected')
				return render_template('batchfileupload.html')
			file = request.files.get('file')
			# check if an empty part is submitted, without filename
			if file.filename == '':
				flash('No File Selected')
				return render_template('batchfileupload.html')
			if file and allowed_file(file.filename):
				output_dataframe = data_engineering.data_processing(file)
				output_html = data_engineering.df_to_html(output_dataframe)
			else:
				flash('Invalid File Format, Try Again!')
				error = 'Invalid File Format, Try Again!'
				return render_template('batchfileupload.html')
		gc.collect()
		flash('Download the csv File')
		return render_template('prediction_result.html', tables=[output_html], error=error)
	except Exception as e:
		return(str(e))

@app.route('/ontheflyform/')
def ontheflyform():
	try:
		flash('Please complete the form and hit SUBMIT')
		return render_template('ontheflyform.html')
	except Exception as e:
		return(str(e))

@app.route('/getformprediction/',methods=['POST','GET'])
def get_form_prediction():
	prediction = ''
	try:
		if request.method=='POST':
			form_result = request.form
			form_result_dict = form_result.to_dict(flat=False)
			print('form_result_dict: ', form_result_dict)
			output_dataframe, small_dataframe = data_engineering.form_processing(form_result_dict)
			output_html = data_engineering.df_to_html(output_dataframe)
			small_html = data_engineering.df_to_html(small_dataframe)
		gc.collect()
		flash('Following are the predictions')
		return render_template('onfly_result.html', tables=[small_html, output_html])
	except Exception as e:
		return (str(e))

#############################---RESTful API---###################################

@auth.get_password
def get_password(email):
	x = c.execute('''SELECT * FROM users WHERE email = %s''', (email, ))
	if int(x) > 0:
		return 'python'
	else:
		return None

@auth.error_handler
def unauthorized():
    return make_response(jsonify({'ERROR': 'Unauthorized Access'}), 401)

@app.route('/magneto/api/v1.0/predict', methods=['POST'])
@auth.login_required
def restful_predict():
    if not request.json or not 'data' in request.json:
        abort(400)
    data_dict = request.json['data']
    dict_for_df = {}
    for key, value in data_dict.items():
    	dict_for_df[key] = [value]
    output_dataframe, small_dataframe = data_engineering.form_processing(dict_for_df)
    output_dict = output_dataframe.to_dict(orient='records')[0]
    return jsonify({'prediction': output_dict}), 201

#################################################################################

if __name__ == '__main__':
	app.run()