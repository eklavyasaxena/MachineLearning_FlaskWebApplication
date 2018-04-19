# Use the basic Python 3 image as launching point
FROM python:3.6.3

# Add the requirements file
ADD requirements.txt /home

# Install required Libraries
RUN pip install -r ./home/requirements.txt

# Add the script
ADD model_development_script.py /home