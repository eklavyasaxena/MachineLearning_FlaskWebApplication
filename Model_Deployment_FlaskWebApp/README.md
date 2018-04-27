# Machine Learning Model Deployment using Flask Web Application

## Summary:
This a Northeastern Academic Project initiative, with following tasks under consideration:
> 1. Create a Web application using Flask that uses the models created (in Pickle format) and stored models on S3
> 2. Build a web page which takes user inputs. The application should allow submission of data for prediction via Forms as well as REST Api calls using JSON
> 3. The application should allow submission on single record of data as well as batch of records data upload to get single/bulk responses
> 4. The Result should be provided to the user as a csv file and a table (with results) should be displayed
> 5. Use the models saved in S3 to run the models

This web app has been setup by primarily following https://pythonprogramming.net/practical-flask-introduction/, and other model deployment examples available online.

## Application Detailing:
> 1. Setup DigitalOcean Server (Use referral https://m.do.co/c/48bcc6762c20 to earn $10 instant credit) with Ubuntu 16.04 with 1 vCPUs and 1GiB Memory and 25GiB Storage
> 2. Download puTTy and WinSCP
> 3. Once in the server, start with an update and upgrade:
```bash
sudo apt-get update && sudo apt-get dist-upgrade
```
> 4. Install mysql as primary database:
```bash
sudo apt-get install apache2 mysql-client mysql-server
```
> 5. Install Apache webserver for Python 3.6:
```bash
sudo apt-get install apache2 apache2-dev
```
> 6. Enable wsgi:
```bash
sudo a2enmod wsgi
```
> 7. Create required folders (not necessary to be followed):
```bash
cd /var/www
sudo mkdir FlaskApp
cd FlaskApp
sudo mkdir FlaskApp
cd FlaskApp
sudo mkdir static
sudo mkdir templates
```
> 8. Create & edit a python file in Nano Editor (optional):
```bash
sudo nano __init__.py
```
> 9. Install & update python pip module:
```bash
sudo apt-get install python-pip
pip install --upgrade pip
```
> 10. Install Virtual Env package:
```bash
sudo pip install virtualenv
```
> 11. Create  Virtual Env to segregate it from WebServer and for Flask to run Python and application in it:
```bash
sudo virtualenv venv
```
> 12. Activate the virtual environment:
```bash
source venv/bin/activate
```
> 13. Install Flask in the VirtualEnv:
```bash
sudo pip install Flask
```
> 14. Deactivate the virtual environment:
```bash
deactivate
```
> 15. Setup the Flask Conf File
```bash
sudo nano /etc/apache2/sites-available/FlaskApp.conf
```
>> Copy Paste the following:
```
<VirtualHost *:80>
                ServerName yourservername
                ServerAdmin youremailid@gmail.com
                WSGIScriptAlias / /var/www/FlaskApp/FlaskApp.wsgi
                <Directory /var/www/FlaskApp/FlaskApp/>
                        Order allow,deny
                        Allow from all
                </Directory>
                Alias /static /var/www/FlaskApp/FlaskApp/static
                <Directory /var/www/FlaskApp/FlaskApp/static/>
                        Order allow,deny
                        Allow from all
                </Directory>
                ErrorLog ${APACHE_LOG_DIR}/error.log
                LogLevel warn
                CustomLog ${APACHE_LOG_DIR}/access.log combined
</VirtualHost>
```
> 16. Enable the Server
```bash
sudo a2ensite FlaskApp
service apache2 reload
```
> 17. Configure the WSGI File:
```bash
cd /var/www/FlaskApp
sudo nano Flaskapp.wsgi
```
>>Copy Paste the following:
```
#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/FlaskApp/")

from FlaskApp import app as application
application.secret_key = '<any_random_gibberish'
```
> 18. To interact with a MySQL database:
```bash
sudo apt-get install python-mysqldb
```
> 19. To install Flask built in forms module called WTForms:
```bash
pip install flask-wtf
```
> 20. Install passlib for password encryption:
```bash
pip install passlib
```
> 21. To email from within the app:
```bash
sudo pip install Flask-Mail
```
> 22. To create SVG (Scalable Vector Graphics) graphs/charts in a variety of styles:
```bash
sudo pip install pygal
```
> 23. To run machine_learning model, following libraries were installed:
```bash
pip install pandas
pip install sklearn
pip install scipy --no-cache-dir
```
> 24. For pickle installing dill package:
```bash
pip install dill
```
> 25. For url requests:
```bash
sudo pip install requests
```
> 26. For dask:
```bash
sudo pip install "dask[complete]"
```