
from flask import Flask, render_template, request
import joblib
import pandas as pd
import logging
from transformers import FeatureCreator

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[logging.FileHandler("flask_app.log"),
                              logging.StreamHandler()])

# Load the trained model, preprocessor, and label encoder
try:
    model = joblib.load('recommendation_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    app.logger.info("Model, preprocessor, and label encoder loaded successfully.")
except FileNotFoundError as e:
    app.logger.error(f"Error loading model files: {e}")
    # Handle the error gracefully, maybe exit or use a fallback
    model = None
    preprocessor = None
    label_encoder = None

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles the recommendation request."""
    if not all([model, preprocessor, label_encoder]):
        return render_template('index.html', error="Model is not available. Please check server logs.")

    try:
        # Get user input from the form and convert to a DataFrame
        features = {
            'Age': [int(request.form['Age'])],
            'Budget': [int(request.form['Budget'])],
            'Interet': [request.form['Interet']],
            'Duree': [int(request.form['Duree'])],
            'Climat': [request.form['Climat']],
            # The model pipeline requires the destination features, but they are not used for prediction.
            # We add placeholders for them.
            'Continent': ['Unknown'],
            'Cout_de_la_Vie': [0],
            'Type_Destination': ['Unknown']
        }
        input_df = pd.DataFrame(features)
        app.logger.info(f"Received user input: {features}")

        # 1. Preprocess the input using the saved pipeline
        # The pipeline handles feature creation and scaling/encoding
        input_processed = preprocessor.transform(input_df)

        # 2. Make a single prediction
        prediction_encoded = model.predict(input_processed)

        # 3. Decode the prediction to get the destination name
        prediction = label_encoder.inverse_transform(prediction_encoded)

        app.logger.info(f"Prediction successful. Recommended destination: {prediction[0]}")
        return render_template('index.html', recommendation=prediction[0])

    except Exception as e:
        app.logger.error(f"An error occurred during recommendation: {e}", exc_info=True)
        return render_template('index.html', error="An error occurred while getting your recommendation.")

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible outside the container
    app.run(host='0.0.0.0', port=5000)
