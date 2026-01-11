
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import  KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the processed data

df = pd.read_csv("C:\App tourisme\Pfe-tourisme-\processed_data.csv")

# Separate features and target

X = df.drop('Destination', axis=1)

y = df['Destination']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)




# Define a simpler parameter grid for XGBClassifier to reduce training time

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize the XGBClassifier
#xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

# Initialize GridSearchCV with fewer folds
#grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
K = 5  # Number of neighbors for KNN
model = KNN(n_neighbors=K)
# Fit the grid search to the data
model.fit(X_train, y_train_encoded)

# Get the best model
#model = grid_search.best_estimator_

# Make predictions on the test set


y_pred_encoded = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

report = classification_report(y_test_encoded, y_pred_encoded, target_names=le.classes_)

# Save the performance to a file
with open('model_performance.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write("Classification Report:\n")
    f.write(report)


print(report)


# Save the trained model to a file

#joblib.dump(model, 'recommendation_model.joblib')

#print("Model training complete. Model saved to 'recommendation_model.joblib'.")

print(len(y_pred_encoded))


print("Accuracy:" , round(accuracy*100,2) , "%")

