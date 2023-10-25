from flask import Flask, request, jsonify,render_template
from simple_salesforce import Salesforce
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
import numpy as np
from simple_salesforce import Salesforce


app = Flask(__name__)

# Salesforce connection settings
SALESFORCE_USERNAME = 'mburnside@dxdo2023.demo'
SALESFORCE_PASSWORD = 'eB!kes1234'
SALESFORCE_SECURITY_TOKEN = ''
SALESFORCE_CLIENT_ID = '3MVG9n_HvETGhr3A12ndHy2N5Wgl9zZKLVWXoOQfMCRCUgHYo2cUXW53EvmPMOwIfLCcJCegGhbHHIw2pmn1g'
SALESFORCE_CLIENT_SECRET = '0BED54F5CC3E4B25160AF24AB3137D272C017BB6DC4135AA54F005572923F10A'


app = Flask(__name__)

# Connect to Salesforce
sf = Salesforce(username=SALESFORCE_USERNAME, password=SALESFORCE_PASSWORD, security_token=SALESFORCE_SECURITY_TOKEN)

# Load the model later for predictions
loaded_model = load('wines_model.joblib')

@app.route('/')  
def list_wines():
    # Query Salesforce for a list of wines
    query = "SELECT Id, Name,alcohol__c,chlorides__c,citric_acid__c,density__c,fixed_acidity__c,free_sulfur_dioxide__c,pH__c,residual_sugar__c,sulphates__c,total_sulfur_dioxide__c,volatile_acidity__c,quality__c FROM Wines__c order by Name"
    wines = sf.query_all(query)['records']

    return render_template('wine.html', wines=wines)

@app.route('/consent')  
def show_consent():
    print ('in /consent')
    return render_template('consent.html')

@app.route('/channelconsent') 
def show_channelconsent():
    print ('in /channelconsent')
    return render_template('channelconsent.html')

@app.route('/detailedconsent')  
def show_detailedconsent():
    print ('in /detailedconsent')
    return render_template('detailedconsent.html')

@app.route('/individualconsent')  
def show_detailedconsent():
    print ('in /individualconsent')
    return render_template('individualconsent.html')


@app.route('/wines', methods=['POST'])  
def predict_wines():
    data=request.get_json()
    fixed_acidity__c=(data.get('fixed_acidity__c'))
    volatile_acidity__c=(data.get('volatile_acidity__c'))
    citric_acid__c=(data.get('citric_acid__c'))
    residual_sugar__c=(data.get('residual_sugar__c'))
    chlorides__c=(data.get('chlorides__c'))
    free_sulfur_dioxide__c=(data.get('free_sulfur_dioxide__c'))
    total_sulfur_dioxide__c=(data.get('total_sulfur_dioxide__c'))
    density__c=(data.get('density__c'))
    pH__c=(data.get('pH__c'))
    sulphates__c=(data.get('sulphates__c'))
    alcohol__c=(data.get('alcohol__c'))

    #input_data = (6.6,0.32,0.44,2.4,0.061,24.0,34.0,0.9978,3.35,0.8,11.6)
    input_data = (fixed_acidity__c,volatile_acidity__c,citric_acid__c,residual_sugar__c,chlorides__c,free_sulfur_dioxide__c,total_sulfur_dioxide__c,density__c,pH__c,sulphates__c,alcohol__c)

    # changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the data as we are predicting the label for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==1):
        result='Good Quality Wine';
        print (result)
    else:
        result='Bad Quality Wine';
        print (result)
        
    return jsonify({'result':result})

    #return render_template('wineresult.html', result=result)


@app.route('/linear_regression')
def linear_regression():
    # Query Salesforce for data
    query = "SELECT AnnualRevenue, NumberOfEmployees FROM wine WHERE AnnualRevenue !=NULL AND NumberOfEmployees !=NULL"
    results = sf.query_all(query)['records']

# Convert data to a Pandas DataFrame
    data = pd.DataFrame(results)
    print (data)

# Prepare data for regression
    X = data['NumberOfEmployees'].values.reshape(-1, 1)
    y = data['AnnualRevenue'].values

# Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

# Make predictions
    y_pred = model.predict(X_test)

# Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    # Create and train a linear regression model (as in the previous code)
    # ...
    # Replace this with your model creation and evaluation code.

    # Render the results template with the model's results
    return render_template('results.html', mse=mse, r2=r2)


port = int(os.environ.get('PORT', 5002))
if __name__ == '__main__':
    app.run(port=port, debug=True)
