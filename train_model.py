import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('car data.csv')

# Feature Engineering: Calculate Car Age

current_year = 2024
df['Age'] = current_year - df['Year']

# Data Preprocessing 
# We map text values to numbers so the model can understand them.

# Fuel_Type: Petrol=0, Diesel=1, CNG=2
df['Fuel_Type'] = df['Fuel_Type'].replace({'Petrol': 0, 'Diesel': 1, 'CNG': 2})

# Seller_Type: Dealer=0, Individual=1
df['Seller_Type'] = df['Seller_Type'].replace({'Dealer': 0, 'Individual': 1})

# Transmission: Manual=0, Automatic=1
df['Transmission'] = df['Transmission'].replace({'Manual': 0, 'Automatic': 1})

# Select Features and Target
X = df[['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Age']]
y = df['Selling_Price']

# Train the Model
# Using Random Forest as it handles this type of data well
model = RandomForestRegressor()
model.fit(X, y)

# Save the model
print("Model trained successfully!")
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved to model.pkl")