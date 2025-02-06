import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("dataModel\stock_price_model.pkl")
scaler = joblib.load("dataModel\scaler.pkl")

def predict_stock_price(input_data):
    # Ensure input is in the correct shape
    input_data = np.array(input_data).reshape(1, -1, 1)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Convert back to the original price scale
    prediction = scaler.inverse_transform(prediction)
    
    return prediction[0][0]
