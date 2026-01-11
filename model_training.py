
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the datasets
df = pd.read_csv('tourisme_dataset.csv')
destinations_df = pd.read_csv('destinations.csv')

# Merge the datasets
df = pd.merge(df, destinations_df, on='Destination')

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Add new features
df['Budget_per_day'] = df['Budget'] / (df['Duree'] + 1e-6)
df['Budget_Ajuste'] = df['Budget_per_day'] / df['Cout_de_la_Vie']
df['Interet_Continent'] = df['Interet'] + '_' + df['Continent']

# Separate features and target
X = df.drop('Destination', axis=1)
y = df['Destination']

# Identify categorical and numerical features
categorical_features = ['Interet', 'Climat', 'Continent', 'Type_Destination', 'Interet_Continent']
numerical_features = ['Age', 'Budget', 'Duree', 'Budget_per_day', 'Cout_de_la_Vie', 'Budget_Ajuste']

# Create preprocessing pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the data
X_processed = pipeline.fit_transform(X)

# Save the preprocessing pipeline
joblib.dump(pipeline, 'preprocessor.joblib')

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save the label encoder
joblib.dump(le, 'label_encoder.joblib')

# Split the data
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = KNN(n_neighbors=5)
model.fit(X_train, y_train_encoded)

# Evaluate the model
y_pred_encoded = model.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
report = classification_report(y_test_encoded, y_pred_encoded, target_names=le.classes_)

# Save performance
with open('model_performance.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write("Classification Report:\n")
    f.write(report)

print(report)

# Save the trained model
joblib.dump(model, 'recommendation_model.joblib')

print("Model training complete.")
