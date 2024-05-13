
# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Chargement des données d'entraînement et de test
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Séparation des caractéristiques (features) et de la cible dans les données d'entraînement
X_train = train_data.drop(columns=['Exited'])
y_train = train_data['Exited']

# Séparation des caractéristiques dans les données de test
X_test = test_data

# Supprimer les colonnes non numériques
X_train_numeric = X_train.select_dtypes(exclude=['object'])
X_test_numeric = X_test.select_dtypes(exclude=['object'])

# Normalisation des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

# Initialisation du modèle de régression logistique
model = LogisticRegression()

# Entraînement du modèle sur l'ensemble de données d'entraînement complet
model.fit(X_train_scaled, y_train)

# Prédiction des probabilités de désabonnement sur l'ensemble de test
probabilities = model.predict_proba(X_test_scaled)[:, 1]

# Création d'un DataFrame pour les résultats de prédiction avec les identifiants de clients
submission_df = pd.DataFrame({'id': test_data['id'], 'Probability_of_Exited': probabilities})

# Enregistrement des résultats dans un fichier CSV
submission_df.to_csv('submission.csv', index=False)
