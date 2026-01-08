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

def get_destination(row):
    scores = {}
    for dest, profile in destination_profiles.items():
        score = 0
        # Score based on interest
        if row['Interet'] in profile['Interet']:
            score += 5
        # Score based on climate
        if row['Climat'] in profile['Climat']:
            score += 3
        # Score based on age
        if profile['Age'][0] <= row['Age'] <= profile['Age'][1]:
            score += 2
        # Score based on budget
        if profile['Budget'][0] <= row['Budget'] <= profile['Budget'][1]:
            score += 2
        # Score based on duration
        if profile['Duree'][0] <= row['Duree'] <= profile['Duree'][1]:
            score += 1
        scores[dest] = score

    # Add some noise to the scores to make it less deterministic
    scores_with_noise = {dest: score + np.random.normal(0, 0.5) for dest, score in scores.items()}

    # Return the destination with the highest score
    return max(scores_with_noise, key=scores_with_noise.get)


# Generate synthetic data
data = {
    'Age': np.random.randint(18, 70, size=num_samples),
    'Budget': np.random.randint(500, 5000, size=num_samples),
    'Interet': np.random.choice(['Culture', 'Nature', 'Aventure', 'Plage', 'Ville'], size=num_samples),
    'Duree': np.random.randint(2, 30, size=num_samples),
    'Climat': np.random.choice(['Chaud', 'Froid', 'Tempéré'], size=num_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Generate destinations based on the rules
df['Destination'] = df.apply(get_destination, axis=1)

# Save to CSV
df.to_csv('tourisme_dataset.csv', index=False)

print("Dataset 'tourisme_dataset.csv' created successfully.")
