
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression  # Or any other model

# Sample Training Data (Replace with your actual data)
train_data = {'km': [10000, 20000, 30000],
              'fuel_type': ['Petrol', 'Diesel', 'Petrol'],
              'price': [100000, 150000, 200000]}
df_train = pd.DataFrame(train_data)

# Preprocessing (Same steps as in your training code)
imputer = SimpleImputer(strategy='median')
df_train['km'] = imputer.fit_transform(df_train[['km']])

le = LabelEncoder()
df_train['fuel_type'] = le.fit_transform(df_train['fuel_type'])

scaler = MinMaxScaler()
numerical_cols = ['km']  # Add other numerical columns
df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])

# Train a simple model (Replace with your actual model training)
model = LinearRegression()
X_train = df_train.drop('price', axis=1)
y_train = df_train['price']
model.fit(X_train, y_train)

# --- Prediction Function ---
def predict_price(km, fuel_type):
    input_data = pd.DataFrame({'km': [km], 'fuel_type': [fuel_type]})

    # ***CRITICAL: Apply the EXACT SAME preprocessing***
    input_data['km'] = imputer.transform(input_data[['km']])  # Imputer must be fitted before
    input_data['fuel_type'] = le.transform([fuel_type])[0]  # Label Encoder must be fitted before
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols]) #Scaler must be fitted before

    prediction = model.predict(input_data)
    return prediction[0]

# --- Example Usage ---
km_input = 15000
fuel_type_input = 'Diesel'

predicted_price = predict_price(km_input, fuel_type_input)
print(f"Predicted Price: {predicted_price}")


# --- Streamlit Integration (Example) ---
import streamlit as st

st.title("Car Price Prediction (Simplified)")

km = st.number_input("Kilometers Driven")
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel'])  # Use unique values from training data

if st.button("Predict"):
    predicted_price = predict_price(km, fuel_type)
    st.write(f"Predicted Price: {predicted_price}")