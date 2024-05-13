import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score

# Charger le jeu de données Iris
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # Nous sélectionnons uniquement les caractéristiques de la longueur et de la largeur des pétales
y = iris.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Liste des valeurs de k à tester
k_values = range(1, 11)

# Stockage des scores de validation croisée pour chaque valeur de k
cv_scores = []

# Boucle sur les valeurs de k
for k in k_values:
    # Créer un objet KMeans avec k clusters
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Calculer les scores de validation croisée sur l'ensemble d'entraînement
    scores = cross_val_score(kmeans, X_train, y_train, cv=5)

    # Stocker la moyenne des scores de validation croisée pour cette valeur de k
    cv_scores.append(scores.mean())

# Tracer les scores de validation croisée en fonction de k
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Cross-validated score')
plt.title('Cross-validation scores for different values of k')
plt.grid(True)
plt.show()

# Choisir la meilleure valeur de k avec le score de validation croisée le plus élevé
best_k = k_values[np.argmax(cv_scores)]
print("Meilleure valeur de k :", best_k)
