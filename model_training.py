
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.neighbors import  KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the processed data
df = pd.read_csv("processed_data.csv")

# Separate features and target
X = df.drop('Destination', axis=1)
y = df['Destination']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best model
model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save the performance to a file
with open('model_performance.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"Accuracy: {accuracy}")
print(report)


# Save the trained model to a file
joblib.dump(model, 'recommendation_model.joblib')

print("Model training complete. Model saved to 'recommendation_model.joblib'.")


