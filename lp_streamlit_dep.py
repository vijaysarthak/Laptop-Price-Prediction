import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the dataset and pipeline/model from pickle files
data = pickle.load(open('data.pkl', 'rb'))  # Dataset
model = pickle.load(open('pipe.pkl', 'rb'))  # Pipeline/model

# Extract unique values for dropdowns from the dataset
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

# Title and description
st.title("Laptop Price Prediction")
st.write("This app predicts laptop prices based on their specifications.")

# Sidebar inputs
st.sidebar.header("Input Features")

# Dropdowns for categorical features
input_data = {}
for feature, options in dropdown_options.items():
    input_data[feature] = st.sidebar.selectbox(feature, options)

# Inputs for numerical features
input_data['Inches'] = st.sidebar.slider('Screen Size (Inches)', min_value=10.0, max_value=18.0, step=0.1)
input_data['Ram'] = st.sidebar.selectbox('RAM (GB)', [4, 8, 16, 32, 64])
input_data['Weight'] = st.sidebar.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
input_data['ScreenW'] = st.sidebar.number_input('Screen Width (pixels)', min_value=800, max_value=4000, step=100)
input_data['ScreenH'] = st.sidebar.number_input('Screen Height (pixels)', min_value=600, max_value=3000, step=100)
input_data['CPU_freq'] = st.sidebar.number_input('CPU Frequency (GHz)', min_value=1.0, max_value=5.0, step=0.1)
input_data['PrimaryStorage'] = st.sidebar.number_input('Primary Storage (GB)', min_value=0, max_value=2048, step=128)
input_data['SecondaryStorage'] = st.sidebar.number_input('Secondary Storage (GB)', min_value=0, max_value=2048, step=128)

# Process inputs for prediction
input_df = pd.DataFrame([input_data])  # Convert input data to a DataFrame

# Prediction button
if st.button('Predict Price'):
    # Log transformation is applied to target; exponentiate the prediction to get the actual price
    log_price = model.predict(input_df)
    actual_price = np.exp(log_price[0])
    
    # Display the prediction
    st.success(f"The predicted price of the laptop is: €{actual_price:.2f}")
