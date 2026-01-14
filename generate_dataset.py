import pandas as pd
import numpy as np

# Charger les destinations à partir du fichier CSV
try:
    destinations_df = pd.read_csv('destinations.csv')
except FileNotFoundError:
    print("Erreur : Le fichier 'destinations.csv' est introuvable.")
    exit()

# Enrichir les destinations avec des données climatiques (absentes du fichier initial)
climate_map = {
    'Paris': 'Tempéré', 'Barcelone': 'Chaud', 'Londres': 'Froid', 'Rome': 'Chaud',
    'Amsterdam': 'Froid', 'Prague': 'Froid', 'Santorin': 'Chaud', 'New York': 'Tempéré',
    'Los Angeles': 'Chaud', 'Vancouver': 'Froid', 'Cancun': 'Chaud', 'Tokyo': 'Tempéré',
    'Kyoto': 'Tempéré', 'Bali': 'Chaud', 'Bangkok': 'Chaud', 'Dubai': 'Chaud',
    'Sydney': 'Chaud', 'Queenstown': 'Froid', 'Bora Bora': 'Chaud', 'Le Caire': 'Chaud',
    'Marrakech': 'Chaud', 'Le Cap': 'Chaud', 'Zanzibar': 'Chaud', 'Rio de Janeiro': 'Chaud',
    'Machu Picchu': 'Froid', 'Buenos Aires': 'Tempéré', 'Patagonie': 'Froid',
    'Amazonie': 'Chaud', 'Iles Galapagos': 'Chaud'
}
destinations_df['Climat'] = destinations_df['Destination'].map(climate_map)

# Définir le nombre d'échantillons à générer
num_samples = 10000

# Extraire les caractéristiques uniques pour la génération
interests = destinations_df['Type_Destination'].unique().tolist()
climates = destinations_df['Climat'].unique().tolist()

def generate_realistic_user_profile():
    """Génère un profil utilisateur avec des distributions plus réalistes."""
    # Distribution d'âge normale centrée autour de 38 ans
    age = int(np.random.normal(38, 12))
    age = np.clip(age, 18, 75)

    # Budget corrélé à l'âge avec une part d'aléa
    base_budget = 1200 + (age - 18) * 60
    budget = int(np.random.normal(base_budget, base_budget * 0.4))
    budget = np.clip(budget, 500, 20000)

    # Durée du séjour avec une distribution qui favorise les séjours plus courts
    duree = int(np.random.lognormal(2, 0.5))
    duree = np.clip(duree, 3, 45)

    return {
        'Age': age,
        'Budget': budget,
        'Interet': np.random.choice(interests),
        'Duree': duree,
        'Climat': np.random.choice(climates)
    }

def get_destination_weighted(user_profile, destinations_df):
    """Calcule les scores de compatibilité et choisit une destination de manière pondérée."""
    scores = {}
    for _, dest in destinations_df.iterrows():
        score = 1.0  # Score de base pour éviter les probabilités nulles

        # Correspondance d'intérêt (haute importance)
        if user_profile['Interet'] == dest['Type_Destination']:
            score += 15
        # Correspondance climatique (moyenne importance)
        if user_profile['Climat'] == dest['Climat']:
            score += 7
        # Proximité du budget (basse importance)
        budget_per_day = user_profile['Budget'] / user_profile['Duree']
        cost_per_day = dest['Cout_de_la_Vie'] * 150  # Facteur de conversion
        budget_diff = abs(budget_per_day - cost_per_day)
        score += max(0, 10 - budget_diff / 100)

        scores[dest['Destination']] = score

    # Sélectionner une destination en utilisant les scores comme poids
    dest_names = list(scores.keys())
    dest_weights = np.array(list(scores.values()))
    dest_weights **= 2 # Exagérer les différences pour favoriser les meilleurs scores
    dest_weights /= dest_weights.sum()

    return np.random.choice(dest_names, p=dest_weights)

# Générer les données
data = []
for _ in range(num_samples):
    user = generate_realistic_user_profile()
    destination = get_destination_weighted(user, destinations_df)
    user['Destination'] = destination
    data.append(user)

# Créer un DataFrame, supprimer les doublons et sauvegarder
df = pd.DataFrame(data)
df.drop_duplicates(inplace=True)
df.to_csv('tourisme_dataset.csv', index=False)

print(f"Généré {len(df)} profils de voyage uniques et cohérents.")
print("Le jeu de données 'tourisme_dataset.csv' a été créé avec succès.")
