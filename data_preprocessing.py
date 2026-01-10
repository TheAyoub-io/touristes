
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the datasets
df = pd.read_csv('tourisme_dataset.csv')
destinations_df = pd.read_csv('destinations.csv')

# Merge the datasets
df = pd.merge(df, destinations_df, on='Destination')

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Add the new feature
df['Budget_per_day'] = df['Budget'] / (df['Duree'] + 1e-6)

# Create advanced interaction features
df['Budget_Ajuste'] = df['Budget_per_day'] / df['Cout_de_la_Vie']
df['Interet_Continent'] = df['Interet'] + '_' + df['Continent']

# Separate features and target
X = df.drop('Destination', axis=1)
y = df['Destination']

# Identify categorical and numerical features
categorical_features = ['Interet', 'Climat', 'Continent', 'Type_Destination', 'Interet_Continent']
numerical_features = ['Age', 'Budget', 'Duree', 'Budget_per_day', 'Cout_de_la_Vie', 'Budget_Ajuste']

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

# Convert to dense array before creating DataFrame
X_processed_df = pd.DataFrame(X_processed.toarray(), columns=all_feature_names)

# Combine processed features with the target variable
processed_df = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)

# Save the processed data to a new CSV file
processed_df.to_csv('processed_data.csv', index=False)

# Save the preprocessing pipeline
joblib.dump(pipeline, 'preprocessor.joblib')

print("Data preprocessing complete. Processed data saved to 'processed_data.csv'.")
print("Preprocessing pipeline saved to 'preprocessor.joblib'.")
