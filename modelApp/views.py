import pandas as pd
import numpy as np
import joblib
import os
from django.shortcuts import render

# Load trained ML model and scaler
model = joblib.load("trainedModel\stock_price_model.pkl")
scaler = joblib.load("trainedModel\scaler.pkl")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def predictor(request):
    return render(request, 'stockdata.html')

def formInfo(request):
    selection = request.GET.get("selection")  # Get the selected dataset path

    if selection == "null":
        return render(request, "result.html", {"result": "Please select a valid dataset."})

    try:
        # Load the dataset
        dataset_path = os.path.join(BASE_DIR, selection)
        df = pd.read_csv(dataset_path)

        # Extract and normalize 'Close' prices
        close_prices = df["Close"].values.reshape(-1, 1)
        close_prices = scaler.transform(close_prices)

        # Prepare the last available data for prediction
        input_data = close_prices[-100:].reshape(1, -1, 1)  # Reshape for LSTM model

        # Make prediction
        y_pred = model.predict(input_data)
        y_pred = scaler.inverse_transform(y_pred)  # Convert back to original scale

        return render(request, "result.html", {"result": f"Predicted Price: {y_pred[0][0]:.2f}"})
    except Exception as e:
        return render(request, "result.html", {"result": f"Error: {str(e)}"})
