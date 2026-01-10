
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import  KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the processed data

df = pd.read_csv("C:\PFE\PFE\processed_data.csv")

# Separate features and target

X = df.drop('Destination', axis=1)

y = df['Destination']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the RandomForestClassifier model

model = RandomForestClassifier(n_estimators=100, random_state=42)

#K=5

#model = KNN(n_neighbors=K)

model.fit(X_train, y_train)

# Make predictions on the test set

y_pred = model.predict(X_test)

# Calculate and print the accuracy

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

# Print the classification report

print(classification_report(y_test, y_pred))

# Save the trained model to a file

joblib.dump(model, 'recommendation_model.joblib')

print("Model training complete. Model saved to 'recommendation_model.joblib'.")

