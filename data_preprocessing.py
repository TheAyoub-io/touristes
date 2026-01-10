
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
df = pd.read_csv('tourisme_dataset.csv')

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Add the new feature
df['Budget_per_day'] = df['Budget'] / (df['Duree'] + 1e-6)

# Separate features and target
X = df.drop('Destination', axis=1)
y = df['Destination']

# Identify categorical and numerical features
categorical_features = ['Interet', 'Climat']
numerical_features = ['Age', 'Budget', 'Duree', 'Budget_per_day']

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit the pipeline to the data
pipeline.fit(X)

# Transform the data
X_processed = pipeline.transform(X)

# Get the new column names from the one-hot encoder
ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
# Combine with numerical feature names
all_feature_names = numerical_features + list(ohe_feature_names)

X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

# Combine processed features with the target variable
processed_df = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)

# Save the processed data to a new CSV file
processed_df.to_csv('processed_data.csv', index=False)

# Save the preprocessing pipeline
joblib.dump(pipeline, 'preprocessor.joblib')

print("Data preprocessing complete. Processed data saved to 'processed_data.csv'.")
print("Preprocessing pipeline saved to 'preprocessor.joblib'.")
