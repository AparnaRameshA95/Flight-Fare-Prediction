import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler
from PIL import Image

# Load Frequency encoded mappings
file_path = 'airport_1_encoded.csv'
airport_1_encoded = pd.read_csv(file_path)

file_path = 'airport_2_encoded.csv'
airport_2_encoded = pd.read_csv(file_path)

file_path = 'carrier_lg_encoded.csv'
carrier_lg_encoded = pd.read_csv(file_path)

file_path = 'carrier_low_encoded.csv'
carrier_low_encoded = pd.read_csv(file_path)

# Sorted list of unique_airports
unique_airports = sorted(['ABE', 'ABQ', 'ACK', 'COS', 'DAL', 'DFW', 'PIT', 'HSV', 'ALB',
       'AMA', 'DEN', 'ATL', 'AUS', 'AVL', 'TUS', 'AZA', 'PHX', 'BDL',
       'SEA', 'BHM', 'ELP', 'CAK', 'CLE', 'BNA', 'BOI', 'BOS', 'MHT',
       'PVD', 'BTV', 'BUF', 'BZN', 'BWI', 'DCA', 'IAD', 'MDW', 'ORD',
       'CHS', 'CID', 'CLT', 'CMH', 'LCK', 'STL', 'MYR', 'JAX', 'DTW',
       'DSM', 'HOU', 'IAH', 'MCO', 'ECP', 'VPS', 'EUG', 'EYW', 'FAR',
       'FCA', 'MSP', 'EWR', 'HPN', 'ISP', 'JFK', 'LGA', 'SWF', 'RSW',
       'GSP', 'GRR', 'GSO', 'LAS', 'IND', 'JAC', 'JAN', 'OAK', 'SFO',
       'SJC', 'FLL', 'MIA', 'BUR', 'LAX', 'LGB', 'ONT', 'SNA', 'LIT',
       'SDF', 'CVG', 'SMF', 'PIE', 'TPA', 'MCI', 'SAT', 'MEM', 'OMA',
       'MKE', 'MSN', 'MSY', 'MVY', 'SAN', 'ORF', 'PHF', 'PNS', 'OKC',
       'PDX', 'PHL', 'PSP', 'PWM', 'RDM', 'RDU', 'RNO', 'ROC', 'SLC',
       'SAV', 'SGF', 'SRQ', 'SYR', 'TYS', 'BIS', 'CAE', 'FAT', 'XNA',
       'RIC', 'ACY', 'ASE', 'ATW', 'BGR', 'BIL', 'EGE', 'FNT', 'FWA',
       'HRL', 'PAE', 'CHI', 'DTT', 'DAY', 'BTR', 'NYC', 'TUL', 'EFD',
       'CRP', 'DAB', 'SAC', 'DET', 'MLB', 'AIY', 'GEG', 'LEX', 'TSS',
       'BLI', 'CGX', 'GPT', 'TLH', 'FTW', 'BMI', 'MDT', 'MGM', 'MFR',
       'SBN', 'MKC', 'LAN', 'AGS', 'MOB', 'MOT', 'BHC', 'GYY', 'FMY',
       'JRB', 'ORL', 'CHA', 'WAS', 'PIA', 'ACV', 'WHR', 'GFK', 'USA',
       'HFD', 'LBE', 'MSO', 'FNL', 'CHO', 'BLV', 'IDA', 'PSC', 'FSD',
       'HHH', 'HTS', 'HVN'])

# Sorted list of unique_carriers
unique_carriers = sorted(['G4', 'DL', 'WN', 'AA', 'UA', 'B6', 'AS', 'F9', 'NK', 'SY', '3M',
       'MX', 'XP', 'US', 'HP', 'CO', 'YX', 'FL', 'NW', 'TW', 'RU', 'DH',
       'J7', 'TZ', 'JI', 'RP', 'P9', 'U5', 'N7', 'NJ', 'QQ', 'WV', 'VX',
       'KW', 'KP', 'ZW', 'UK', 'W7', 'HQ', 'FF', 'TB', 'LC', 'YY', 'PA',
       'YV', '5J', 'KN', '9K', 'E9', 'PN', 'BF', '9N', 'U2', 'OE', 'W9',
       'ZV', 'RL', 'T3', 'OP', 'OO', 'AQ', 'QX', 'OH', 'KS', 'XJ', 'ZA',
       'SX'])

# Load the pre-trained model and the dataframe
model = pickle.load(open('nrf_model.pkl', 'rb'))


# Define a function to preprocess the input data
def preprocess_input(input_data):
    
    print("Input in function")
    print(input_data)

    # Log transformation for 'passengers', 'fare_lg', and 'fare_low'
    for column in ['passengers','fare_lg', 'fare_low']:
        # Apply log(1 + x) transformation to handle zero values safely
        input_data[column] = np.log1p(input_data[column]).astype(float)
    print("Input after log transform")
    print(input_data)

    # Robust_scaling
    # Parameters for robust_scaling
    Feature = ['nsmiles', 'passengers', 'large_ms', 'fare_lg', 'lf_ms', 'fare_low']
    IQR = [1109.000000, 2.702236, 0.392100, 0.487631, 0.592000, 0.493406]
    Median = [1021.000000, 4.744932, 0.652800, 5.342526, 0.360000, 5.207462]
    # Create DataFrame for IQR
    df_iqr = pd.DataFrame([IQR], columns=Feature)
    # Create DataFrame for Median
    df_median = pd.DataFrame([Median], columns=Feature)
    robust_scaling_columns = ['nsmiles', 'large_ms',  'lf_ms',  'passengers',   'fare_lg',  'fare_low']
    # Initialize RobustScaler
    robust_scaler = RobustScaler()
    # Fit and transform the selected numerical columns
    numerical_columns = [col for col in robust_scaling_columns]
    input_data[numerical_columns] = (input_data[numerical_columns]-df_median[numerical_columns])/df_iqr[numerical_columns]
    print("Input after robust scaling")
    print(input_data)

    # Frequency_Encoding
    # List of columns to be frequency encoded
    columns_to_encode = ['airport_1','airport_2','carrier_lg','carrier_low']
    # Apply Frequency Encoding
    for col in columns_to_encode:
        if col == 'airport_1':
            input_data[col] = airport_1_encoded[input_data[col]]
        elif col == 'airport_2':
            input_data[col] = airport_2_encoded[input_data[col]]
        elif col == 'carrier_lg':
            input_data[col] = carrier_lg_encoded[input_data[col]]
        elif col == 'carrier_low':
            input_data[col] = carrier_low_encoded[input_data[col]]
    
    print("Input after encoding")
    print(input_data)
    return input_data

# Create the Streamlit app
st.title("US Airline Fare Prediction")

# Load the image
image_path = "airline_3.png"  # Replace with your image file path
image = Image.open(image_path)

# Display the image below the title
st.image(image, use_column_width=True)

# Create input fields for the features
st.sidebar.header("Enter Flight Details")

# User input for each feature
Year = st.sidebar.number_input("Year:", min_value=1994, max_value=2030, value=1994)
quarter = st.sidebar.number_input("Quarter (1-4):", min_value=1, max_value=4, value=1)
passengers = st.sidebar.number_input("Number of Passengers:",value=00.0)
airport_1 = st.sidebar.selectbox("Departure Airport:", unique_airports)
airport_2 = st.sidebar.selectbox("Arrival Airport:", unique_airports)
nsmiles = st.sidebar.number_input("Distance(miles):",value=0)
# Updated fare input with dollar symbol formatting
carrier_lg = st.sidebar.selectbox("Large Carrier:", unique_carriers)
large_ms = st.sidebar.number_input("Large Fare Market Share(%):",value=00.0)
fare_lg = st.sidebar.number_input("Fare for Large Carrier ($):", value=00.0, format="%.2f")
# Select box inputs for airports and carriers, sorted alphabetically
carrier_low = st.sidebar.selectbox("Low Carrier:", unique_carriers)
lf_ms = st.sidebar.number_input("Low Fare Market Share(%):",value=00.0)
fare_low = st.sidebar.number_input("Fare for Low Carrier ($):", value=00.0, format="%.2f")


# Create a button to predict the fare
if st.sidebar.button("Predict Fare"):
    user_data = {
        'Year': Year,
        'quarter': quarter,
        'nsmiles': nsmiles,
        'large_ms': large_ms,
        'lf_ms': lf_ms,
        'passengers': passengers,
        'fare_lg': fare_lg,
        'fare_low': fare_low,
        "airport_1": airport_1,
        "airport_2": airport_2,
        "carrier_lg": carrier_lg,
        "carrier_low": carrier_low
    }
    input_data = pd.DataFrame([user_data])
    print("Entered input")
    print(input_data)
    
    # Preprocess the input data
    processed_input = preprocess_input(input_data)
    print("Input data after processing")
    print(processed_input)
    
    # Make the prediction
    predicted_fare = model.predict(processed_input)

    # Display the prediction (reverse the log transformation)
    st.write(f"## Predicted Fare: ${np.expm1(predicted_fare[0]):.2f}")
