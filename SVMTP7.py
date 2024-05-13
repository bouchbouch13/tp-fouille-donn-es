# Import des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Chargement des données
data = pd.read_csv('student_admission_dataset.csv')

# Filtrage pour ne garder que les statuts "Rejected" et "Accepted"
data = data[data['Admission_Status'].isin(['Rejected', 'Accepted'])]

# Séparation des caractéristiques (features) et de la variable cible
X = data[['GPA', 'SAT_Score', 'Extracurricular_Activities']]
y = data['Admission_Status']

# Conversion de la variable cible en valeurs numériques (0 pour Rejected, 1 pour Accepted)
y = y.replace({'Rejected': 0, 'Accepted': 1})

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des caractéristiques (features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création du modèle SVM avec le noyau linéaire et la spécification de la décision multiclasse
svm_model = SVC(kernel='linear')

# Entraînement du modèle
svm_model.fit(X_train_scaled, y_train)

# Prédiction sur l'ensemble de test
y_pred = svm_model.predict(X_test_scaled)

# Évaluation du modèle
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Représentation des résultats avec les régions de décision
# Visualisation de la prédiction par rapport à la variable cible
plt.figure(figsize=(8, 6))

# Affichage des régions de décision
h = .02  # step size in the mesh
x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape)]) # ajouter une troisième colonne de zéros pour satisfaire l'attente du modèle

# Mettre le résultat dans un graphique coloré
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)

# Affichage des points de test
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred, cmap='coolwarm', marker='o', edgecolors='k')
plt.xlabel('GPA')
plt.ylabel('SAT Score')
plt.title('Student Admission Prediction with Decision Regions')
plt.colorbar(label='Predicted Admission Status')
plt.grid(True)
plt.show()
