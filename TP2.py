from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Charger les données sur le cancer du sein
cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle de régression logistique
model = LogisticRegression(max_iter=3000, solver='lbfgs')

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
predictions = model.predict(X_test)

# Calculer l'exactitude du modèle
accuracy = accuracy_score(y_test, predictions)
print("Exactitude du modèle : {:.2f}".format(accuracy))

# Afficher le rapport de classification
report = classification_report(y_test, predictions, target_names=cancer_data.target_names)
print("Rapport de classification :\n", report)

# Affichage de l'exactitude sous forme de diagramme
plt.bar(['Exactitude'], [accuracy])
plt.ylim(0, 1)  # Limite l'axe y de 0 à 1
plt.ylabel('Exactitude')
plt.title('Exactitude du modèle')
plt.show()

# Affichage de la relation entre "radius_mean" et "concave points_mean"
radius_mean = X_test[:, cancer_data.feature_names.tolist().index('mean radius')]
concave_points_mean = X_test[:, cancer_data.feature_names.tolist().index('mean concave points')]

plt.figure(figsize=(8, 6))
plt.scatter(radius_mean[y_test == 0], concave_points_mean[y_test == 0], color='blue', label='Bénin')
plt.scatter(radius_mean[y_test == 1], concave_points_mean[y_test == 1], color='red', label='Maligne')
plt.xlabel('Radius Mean')
plt.ylabel('Concave Points Mean')
plt.title('Relation entre Radius Mean et Concave Points Mean')
plt.legend()
plt.grid(True)
plt.show()
