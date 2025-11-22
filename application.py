from flask import Flask, render_template, request
import pickle
import numpy as np

application = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from form
            Present_Price = float(request.form['Present_Price'])
            Kms_Driven = int(request.form['Kms_Driven'])
            Owner = int(request.form['Owner'])
            Year = int(request.form['Year'])
            
            # Calculate Age
            Age = 2024 - Year

            # Handle Categorical Inputs (Must match train_model.py mappings)
            
            # Fuel Type
            Fuel_Type_Input = request.form['Fuel_Type']
            if Fuel_Type_Input == 'Petrol':
                Fuel_Type = 0
            elif Fuel_Type_Input == 'Diesel':
                Fuel_Type = 1
            else:
                Fuel_Type = 2 # CNG

            # Seller Type
            Seller_Type_Input = request.form['Seller_Type']
            if Seller_Type_Input == 'Dealer':
                Seller_Type = 0
            else:
                Seller_Type = 1 # Individual

            # Transmission
            Transmission_Input = request.form['Transmission']
            if Transmission_Input == 'Manual':
                Transmission = 0
            else:
                Transmission = 1 # Automatic

            # Prepare input array for the model
            features = np.array([[Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner, Age]])
            
            # Make Prediction
            prediction = model.predict(features)
            
            # The model returns price in Lakhs
            output = round(prediction[0], 2)

            return render_template('index.html', prediction_text=f'Estimated Car Price: ₹{output} Lakhs')

        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    application.run(debug=True)