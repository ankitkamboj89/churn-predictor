from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load("../model/model.pkl")

scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = {}
        for key in request.form:
            input_data[key] = request.form[key]

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical
        for col, le in label_encoders.items():
            input_df[col] = le.transform(input_df[col])

        # Scale numeric
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        return render_template("result.html", prediction=prediction, probability=round(prob * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
