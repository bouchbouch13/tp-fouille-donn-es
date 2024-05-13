import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Charger le jeu de données Iris
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # Nous sélectionnons uniquement les caractéristiques de la longueur et de la largeur des pétales
y = iris.target

# Créer un objet KMeans
kmeans = KMeans(n_clusters=3, random_state=42)

# Adapter le modèle aux données
kmeans.fit(X)

# Effectuer des prédictions sur les données d'entraînement
y_pred = kmeans.predict(X)

# Afficher les clusters et les centres des clusters
plt.figure(figsize=(8, 6))

# Plot des points de données
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.7, label='Data Points')

# Plot des centres des clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Cluster Centers')

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Clusters of Iris flowers based on Petal Length vs. Petal Width')
plt.legend()
plt.grid(True)
plt.show()
