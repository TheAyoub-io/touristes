import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 5000

# Define destination profiles
destination_profiles = {
    'Paris': {
        'Interet': ['Culture', 'Ville'], 'Climat': ['Tempéré'],
        'Age': (25, 60), 'Budget': (1500, 4000), 'Duree': (5, 15)
    },
    'Tokyo': {
        'Interet': ['Culture', 'Ville', 'Aventure'], 'Climat': ['Tempéré'],
        'Age': (20, 50), 'Budget': (3000, 10000), 'Duree': (7, 20)
    },
    'New York': {
        'Interet': ['Ville', 'Aventure'], 'Climat': ['Tempéré', 'Froid'],
        'Age': (20, 55), 'Budget': (2500, 10000), 'Duree': (4, 10)
    },
    'Bali': {
        'Interet': ['Nature', 'Plage'], 'Climat': ['Chaud'],
        'Age': (18, 40), 'Budget': (500, 2500), 'Duree': (7, 21)
    },
    'Rome': {
        'Interet': ['Culture', 'Ville'], 'Climat': ['Tempéré', 'Chaud'],
        'Age': (25, 65), 'Budget': (1000, 3500), 'Duree': (4, 14)
    },
    'Le Caire': {
        'Interet': ['Culture'], 'Climat': ['Chaud'],
        'Age': (30, 70), 'Budget': (800, 2000), 'Duree': (6, 12)
    },
    'Rio de Janeiro': {
        'Interet': ['Nature', 'Plage', 'Aventure'], 'Climat': ['Chaud'],
        'Age': (18, 45), 'Budget': (700, 3000), 'Duree': (5, 15)
    },
    'Sydney': {
        'Interet': ['Nature', 'Plage', 'Ville'], 'Climat': ['Tempéré', 'Chaud'],
        'Age': (20, 50), 'Budget': (2000, 4500), 'Duree': (10, 25)
    },
    'Barcelone': {
        'Interet': ['Culture', 'Plage', 'Ville'], 'Climat': ['Tempéré', 'Chaud'],
        'Age': (20, 50), 'Budget': (800, 3000), 'Duree': (3, 10)
    },
    'Londres': {
        'Interet': ['Culture', 'Ville'], 'Climat': ['Tempéré', 'Froid'],
        'Age': (25, 60), 'Budget': (1500, 4000), 'Duree': (4, 12)
    }
}

# Generate data from destination profiles
all_data = []
for destination, profile in destination_profiles.items():
    # Add a bit of noise to make the data more robust
    age_min, age_max = profile['Age']
    budget_min, budget_max = profile['Budget']
    duree_min, duree_max = profile['Duree']

    data = {
        'Age': np.random.randint(age_min, age_max, size=num_samples),
        'Budget': np.random.randint(budget_min, budget_max, size=num_samples),
        'Interet': np.random.choice(profile['Interet'], size=num_samples),
        'Duree': np.random.randint(duree_min, duree_max, size=num_samples),
        'Climat': np.random.choice(profile['Climat'], size=num_samples)
    }

    df_dest = pd.DataFrame(data)
    df_dest['Destination'] = destination
    all_data.append(df_dest)

# Combine all profile data into a single DataFrame
df = pd.concat(all_data, ignore_index=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Save to CSV
df.to_csv('tourisme_dataset.csv', index=False)

print("Dataset 'tourisme_dataset.csv' created successfully.")
