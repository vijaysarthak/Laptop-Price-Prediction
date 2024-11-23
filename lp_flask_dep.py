from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

lppred = Flask(__name__)

# Load the dataset and trained model
data = pickle.load(open('data.pkl', 'rb'))
model = pickle.load(open('pipe.pkl', 'rb'))

# Extract unique values for dropdown options from the dataset
dropdown_options = {
    'Company': sorted(data['Company'].unique()),
    'Product': sorted(data['Product'].unique()),
    'TypeName': sorted(data['TypeName'].unique()),
    'OS': sorted(data['OS'].unique()),
    'Screen': sorted(data['Screen'].unique()),
    'Touchscreen': ['Yes', 'No'],
    'IPSpanel': ['Yes', 'No'],
    'RetinaDisplay': ['Yes', 'No'],
    'CPU_company': sorted(data['CPU_company'].unique()),
    'CPU_model': sorted(data['CPU_model'].unique()),
    'PrimaryStorageType': sorted(data['PrimaryStorageType'].unique()),
    'SecondaryStorageType': sorted(data['SecondaryStorageType'].unique()),
    'GPU_company': sorted(data['GPU_company'].unique()),
    'GPU_model': sorted(data['GPU_model'].unique())
}

# For numerical columns, we'll generate a dropdown range based on the dataset
numeric_columns = ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage']

numeric_dropdown_options = {}
for col in numeric_columns:
    # For simplicity, we'll create dropdowns for discrete numeric values
    numeric_dropdown_options[col] = sorted(data[col].dropna().unique().tolist())

@lppred.route('/')
def home():
    # Render the form with dropdown options
    return render_template('index.html', dropdown_options=dropdown_options, numeric_dropdown_options=numeric_dropdown_options)

@lppred.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from the form
        inputs = {
            'Company': request.form['Company'],
            'Product': request.form['Product'],
            'TypeName': request.form['TypeName'],
            'OS': request.form['OS'],
            'Screen': request.form['Screen'],
            'Touchscreen': request.form['Touchscreen'],
            'IPSpanel': request.form['IPSpanel'],
            'RetinaDisplay': request.form['RetinaDisplay'],
            'CPU_company': request.form['CPU_company'],
            'CPU_model': request.form['CPU_model'],
            'PrimaryStorageType': request.form['PrimaryStorageType'],
            'SecondaryStorageType': request.form['SecondaryStorageType'],
            'GPU_company': request.form['GPU_company'],
            'GPU_model': request.form['GPU_model'],
            'Inches': float(request.form['Inches']),
            'Ram': int(request.form['Ram']),
            'Weight': float(request.form['Weight']),
            'ScreenW': int(request.form['ScreenW']),
            'ScreenH': int(request.form['ScreenH']),
            'CPU_freq': float(request.form['CPU_freq']),
            'PrimaryStorage': int(request.form['PrimaryStorage']),
            'SecondaryStorage': int(request.form['SecondaryStorage'])
        }

        # Convert inputs to DataFrame for prediction
        input_df = pd.DataFrame([inputs])

        # Predict the log price and exponentiate to get the actual price
        log_price = model.predict(input_df)[0]
        predicted_price = np.exp(log_price)

        # Render the index template with the predicted price
        return render_template('index.html', 
                               dropdown_options=dropdown_options, 
                               numeric_dropdown_options=numeric_dropdown_options, 
                               predicted_price=f'€{predicted_price:.2f}')
    except Exception as e:
        # In case of error, render the index with an error message
        return render_template('index.html', 
                               dropdown_options=dropdown_options, 
                               numeric_dropdown_options=numeric_dropdown_options, 
                               predicted_price=f"Error: {str(e)}")

if __name__ == '__main__':
    lppred.run(debug=True)
