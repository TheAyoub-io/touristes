
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessor
model = joblib.load('recommendation_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')
le = joblib.load('label_encoder.joblib')
destinations_df = pd.read_csv('destinations.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input from the form
    features = {
        'Age': int(request.form['Age']),
        'Budget': int(request.form['Budget']),
        'Interet': request.form['Interet'],
        'Duree': int(request.form['Duree']),
        'Climat': request.form['Climat']
    }

    # Create a DataFrame from the user input
    input_df = pd.DataFrame([features])

    # Add all destinations for each user
    input_df = pd.concat([input_df.assign(Destination=dest) for dest in destinations_df['Destination']], ignore_index=True)

    # Merge with destination features
    input_df = pd.merge(input_df, destinations_df, on='Destination')

    # Add new features
    input_df['Budget_per_day'] = input_df['Budget'] / (input_df['Duree'] + 1e-6)
    input_df['Budget_Ajuste'] = input_df['Budget_per_day'] / input_df['Cout_de_la_Vie']
    input_df['Interet_Continent'] = input_df['Interet'] + '_' + input_df['Continent']

    # Preprocess the user input
    input_processed = preprocessor.transform(input_df)

    # Make a prediction
    prediction_encoded = model.predict_proba(input_processed)

    # Get the best destination
    probabilities = prediction_encoded.diagonal()
    best_destination_index = probabilities.argmax()
    prediction = le.classes_[best_destination_index]


    return render_template('index.html', recommendation=prediction)

if __name__ == '__main__':
    app.run(debug=True)
