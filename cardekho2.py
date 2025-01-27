#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the Excel file
file_path = r'C:\Users\manis\OneDrive\Desktop\final_car.xlsx'  # Provide the correct file path
df = pd.read_excel(file_path)

# Display the first few rows to ensure it is loaded correctly
print(df.head(8370))


# In[2]:


# Check for missing values in the dataset
print(df.isnull().sum())


# In[3]:


import pandas as pd

# Load the dataset
file_path = r'C:\Users\manis\OneDrive\Desktop\final_car.xlsx'  # Change this to your file's path
df = pd.read_excel(file_path)

# Convert 'Mileage' column to numeric, coercing errors to NaN (useful for non-numeric entries)
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')

# Now fill missing values for 'Mileage' with the mean
df['Mileage'] = df['Mileage'].fillna(df['Mileage'].mean())

# Similarly, for other columns, apply the same approach as needed

# Fill missing values for other columns as well
df['bt'] = df['bt'].fillna(df['bt'].mode()[0])  # Fill with mode for 'bt'
df['Engine'] = df['Engine'].fillna(df['Engine'].mean())  # Fill with mean for 'Engine'
df['Seats'] = df['Seats'].fillna(df['Seats'].mode()[0])  # Fill with mode for 'Seats'

# For categorical columns with a lot of missing values, fill with mode or 'Unknown'
df['Adjustable Head Lights'] = df['Adjustable Head Lights'].fillna(df['Adjustable Head Lights'].mode()[0])
df['Air Conditioner'] = df['Air Conditioner'].fillna(df['Air Conditioner'].mode()[0])
df['Heater'] = df['Heater'].fillna(df['Heater'].mode()[0])
df['Power Steering'] = df['Power Steering'].fillna(df['Power Steering'].mode()[0])

# Verify the missing values after handling
print(df.isnull().sum())

# Save the cleaned dataset to a new Excel file
df.to_excel('cleaned_dataset.xlsx', index=False)


# In[4]:


# Step 1: Check the current data types
print(df.dtypes)

# Step 2: Convert data types where necessary

# Convert 'km' and 'price' columns to numeric types (float)
df['km'] = pd.to_numeric(df['km'], errors='coerce')  # Coerce errors to NaN if any issues occur
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Convert 'modelYear' to a datetime or integer type
df['modelYear'] = pd.to_datetime(df['modelYear'], errors='coerce', format='%Y')

# Convert 'Engine' and 'Mileage' to numeric (float) if they aren't already
df['Engine'] = pd.to_numeric(df['Engine'], errors='coerce')
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')

# For categorical columns, ensure they are strings
df['transmission'] = df['transmission'].astype('str')
df['owner'] = df['owner'].astype('str')
df['oem'] = df['oem'].astype('str')
df['model'] = df['model'].astype('str')
df['variantName'] = df['variantName'].astype('str')

# For other columns, you can similarly apply correct formatting
# Check if there are any other columns that need format correction

# Check data types after conversion
print(df.dtypes)


# In[5]:


# Standardize text columns to lowercase
df['transmission'] = df['transmission'].str.lower()
df['owner'] = df['owner'].str.lower()
df['oem'] = df['oem'].str.lower()
df['model'] = df['model'].str.lower()
df['variantName'] = df['variantName'].str.lower()
df.to_csv('cleaned_data.csv', index=False)


# In[6]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Create a copy of the dataframe to avoid changing the original
df_encoded = df.copy()

# One-hot encoding for nominal categorical variables
nominal_columns = ['transmission', 'owner', 'oem', 'model', 'variantName', 'Adjustable Head Lights', 
                   'Air Conditioner', 'Heater', 'Power Steering']  # Add more if needed
df_encoded = pd.get_dummies(df_encoded, columns=nominal_columns, drop_first=True)

# Label encoding or Ordinal encoding for ordinal categorical variables
# Assuming 'ownerNo' is ordinal (1st, 2nd, 3rd, etc.)
label_encoder = LabelEncoder()
df_encoded['ownerNo'] = label_encoder.fit_transform(df_encoded['ownerNo'])

# Check the encoded dataset
df_encoded.head(8370)


# In[7]:


# Impute missing values in 'km' column with the median
df_encoded['km'] = df_encoded['km'].fillna(df_encoded['km'].median())


# In[8]:


# Extract 'year' from 'modelYear'
df_encoded['modelYear'] = pd.to_datetime(df_encoded['modelYear'])
df_encoded['modelYear'] = df_encoded['modelYear'].dt.year  # Extract only the year


# In[9]:


df_encoded.head(8370)


# In[10]:


# Check for duplicate rows
df_encoded.duplicated().sum()  # If the result is greater than 0, there are duplicates


# In[11]:


# Drop duplicate rows
df_encoded_cleaned = df_encoded.drop_duplicates()

# Check if the duplicates were removed
print(f"Number of duplicate rows removed: {df_encoded.shape[0] - df_encoded_cleaned.shape[0]}")


# In[12]:


# Impute missing values in 'km' with the mean of the column
df['km'].fillna(df['km'].mean(), inplace=True)

# Check the updated 'km' column to ensure NaN values are replaced
df['km'].isna().sum()


# In[13]:


from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Select numerical columns for normalization
numerical_columns = ['km', 'price', 'Engine', 'Mileage', 'Seats']

# Fit and transform the data
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Display the transformed data
print(df[numerical_columns].head(8370))


# In[14]:


import pandas as pd

# Function to remove outliers using IQR
def remove_outliers_iqr(df):
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numerical_cols:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        
        # Calculate IQR (Interquartile Range)
        IQR = Q3 - Q1
        
        # Define lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers (values outside the lower and upper bounds)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

# Apply the function to remove outliers
cleaned_data = remove_outliers_iqr(df)

# Check the shape of the cleaned data
print(cleaned_data.shape)



# In[15]:


cleaned_data.head(6198)


# In[16]:


# Descriptive statistics for numerical features
numerical_features = ['km', 'price', 'Engine', 'Mileage', 'Seats']

# Summary statistics
summary_stats = cleaned_data[numerical_features].describe().T
print(summary_stats)

# Mode for numerical columns
modes = cleaned_data[numerical_features].mode()
print("Modes of numerical features:")
print(modes)


# In[17]:


#km:Constant value (0.386974) across all rows suggests that this column is not providing any variation, making it unsuitable for modeling or analysis.
#Seats:Constant value (0.375), indicating no variation and therefore no predictive utility.So, EDA can be done on mileage, engine and price.

import matplotlib.pyplot as plt
import seaborn as sns

# Columns for visualization
visual_features = ['price', 'Engine', 'Mileage']

# Set up the layout for visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram for each feature
for idx, feature in enumerate(visual_features):
    sns.histplot(cleaned_data[feature], kde=True, bins=30, ax=axes[idx // 2, idx % 2], color='skyblue')
    axes[idx // 2, idx % 2].set_title(f'Distribution of {feature}')
    axes[idx // 2, idx % 2].set_xlabel(feature)
    axes[idx // 2, idx % 2].set_ylabel('Frequency')

# Scatterplot to examine correlations
sns.scatterplot(data=cleaned_data, x='Engine', y='price', ax=axes[1, 1], color='green', alpha=0.6)
axes[1, 1].set_title('Engine vs Price')
axes[1, 1].set_xlabel('Engine')
axes[1, 1].set_ylabel('Price')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = cleaned_data[['price', 'Engine', 'Mileage']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()



# In[ ]:


cleaned_data.to_csv('cleaned_data.csv')
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the cleaned dataset
data = pd.read_csv('cleaned_data.csv')  # Replace with your actual file path

# Features and target selection
X = data[['Mileage', 'Engine']]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection: Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_rf_model = grid_search.best_estimator_

# Train the best model on the training data
best_rf_model.fit(X_train, y_train)

# Predictions on test data
y_pred = best_rf_model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Feature importance
feature_importances = best_rf_model.feature_importances_
for feature, importance in zip(X.columns, feature_importances):
    print(f"Feature: {feature}, Importance: {importance}")


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

residuals = y_test - y_pred

# Plot residuals
sns.histplot(residuals, kde=True, bins=30)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


import numpy as np

# Identify potential outliers
threshold = 0.00005  # Define a threshold based on the spread
outliers = residuals[np.abs(residuals) > threshold]
print("Outliers:", outliers)

plt.scatter(y_pred, residuals, alpha=0.7, edgecolors='k')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()


# In[ ]:


import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Assuming `y_test` contains actual test labels and `y_pred` contains predictions
outlier_indices_full = [2687, 4320, 167, 3892, 156, 4529, 4194, 881, 1400, 4957]  # Original dataset indices

# Map these indices to the test set
# Ensure you have a mapping if you split the data
outlier_indices = [idx for idx in outlier_indices_full if idx < len(y_test)]

# Create boolean mask to exclude outliers
mask = np.ones(len(y_test), dtype=bool)
mask[outlier_indices] = False

# Filter y_test and y_pred
y_test_cleaned = y_test[mask]
y_pred_cleaned = y_pred[mask]

# Calculate metrics
rmse_with = np.sqrt(mean_squared_error(y_test, y_pred))
r2_with = r2_score(y_test, y_pred)

rmse_without = np.sqrt(mean_squared_error(y_test_cleaned, y_pred_cleaned))
r2_without = r2_score(y_test_cleaned, y_pred_cleaned)

# Print results
print(f"RMSE With Outliers: {rmse_with}, Without Outliers: {rmse_without}")
print(f"R² With Outliers: {r2_with}, Without Outliers: {r2_without}")

#The presence of these outliers has a minimal impact on the model's performance.


# In[ ]:


import joblib

# Save the trained model to the specified path on the Desktop
model_path = r"C:\Users\manis\OneDrive\Desktop\best_rf_model.joblib"
joblib.dump(best_rf_model, model_path)

print(f"Model saved to {model_path}")


# In[ ]:


import joblib

# Load the trained model from the desktop
model_path = r"C:\Users\manis\OneDrive\Desktop\best_rf_model.joblib"
best_rf_model = joblib.load(model_path)

print("Model loaded successfully.")


# In[ ]:


#get_ipython().system('pip install streamlit')
#get_ipython().system('pip install scikit-learn pandas numpy')


import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model
model_path = r"C:\Users\manis\OneDrive\Desktop\best_rf_model.joblib"
best_rf_model = joblib.load(model_path)

# Function to predict car price
def predict_price(features):
    # Convert features to a numpy array and reshape if necessary
    features_array = np.array(features).reshape(1, -1)
    prediction = best_rf_model.predict(features_array)
    return prediction[0]

# Title of the web app
st.title("Car Price Prediction App")

# Description of the app
st.write("""
    This app predicts the price of a car based on its features.
    Enter the values of the car's features below to get the estimated price.
""")

# Input fields for the car features (example: age, mileage, horsepower, etc.)
# You can customize the features as per your model's requirements
age = st.number_input("Car Age (in years)", min_value=0, max_value=30, step=1)
mileage = st.number_input("Mileage (in km)", min_value=0, max_value=500000, step=1000)
horsepower = st.number_input("Horsepower", min_value=50, max_value=1000, step=10)
brand = st.selectbox("Brand", ["Brand A", "Brand B", "Brand C", "Brand D"])  # Example brands

# Convert categorical input (e.g., brand) into a numerical value if necessary
# Here, we assume a simple encoding: 0, 1, 2, 3 for each brand
brand_mapping = {"Brand A": 0, "Brand B": 1, "Brand C": 2, "Brand D": 3}
brand_value = brand_mapping.get(brand, 0)

# Collect all features into a list
features = [age, mileage, horsepower, brand_value]

# Button to predict
if st.button("Predict Price"):
    # Perform the prediction
    predicted_price = predict_price(features)
    
    # Display the result
    st.write(f"The predicted price of the car is: ${predicted_price:,.2f}")

# Instructions and error handling
st.write("""
    Please enter the car features above and click 'Predict Price' to get the estimated price.
    Ensure that all values are entered correctly to avoid any errors in prediction.
""")














