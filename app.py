from flask import Flask, request, jsonify,render_template
from simple_salesforce import Salesforce,SalesforceLogin
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from joblib import dump, load
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64


app = Flask(__name__)

# Salesforce connection settings
SALESFORCE_USERNAME = 'mburnside@dxdo2023.demo'
SALESFORCE_PASSWORD = 'eB!kes1234'
SALESFORCE_SECURITY_TOKEN = ''
SALESFORCE_CLIENT_ID = '3MVG9n_HvETGhr3A12ndHy2N5Wgl9zZKLVWXoOQfMCRCUgHYo2cUXW53EvmPMOwIfLCcJCegGhbHHIw2pmn1g'
SALESFORCE_CLIENT_SECRET = '0BED54F5CC3E4B25160AF24AB3137D272C017BB6DC4135AA54F005572923F10A'


app = Flask(__name__)

print ('connecting')
# Connect to Salesforce
sf =Salesforce(username=SALESFORCE_USERNAME, password=SALESFORCE_PASSWORD, security_token=SALESFORCE_SECURITY_TOKEN)

print (sf)
accesstoken=sf.session_id
print ('you are connected')
# Load the model later for predictions
loaded_model = load('wines_model310.joblib')
print (loaded_model)

@app.route('/')  
def start():
    return render_template('start.html')

@app.route('/ebikesflow')  
def flow():
    # test lightning out
    return render_template('ebikesflow.html',token=accesstoken)


@app.route('/lodemo')  
def lwc():
    # test lightning out
    return render_template('lodemo.html',token=accesstoken)




port = int(os.environ.get('PORT', 5002))
if __name__ == '__main__':
    app.run(port=port, debug=True)
