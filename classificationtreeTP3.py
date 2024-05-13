# Charger les bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Charger les données
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Filtrer les données pour ne conserver que les valeurs positives de 'mean concave points'
X_filtered = X[['mean radius', 'mean concave points']][X['mean concave points'] >= 0]

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y[y.index.isin(X_filtered.index)], test_size=0.2, random_state=42)

# Créer le modèle d'arbre de décision avec la région de décision CART et une profondeur maximale de 5
model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle:", accuracy)

# Tracer la frontière de décision
x_min, x_max = X_filtered.iloc[:, 0].min() - 1, X_filtered.iloc[:, 0].max() + 1
y_min, y_max = X_filtered.iloc[:, 1].min() - 1, X_filtered.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

# Afficher les zones de décision
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)

# Afficher les points d'entraînement
plt.scatter(X_filtered.iloc[:, 0], X_filtered.iloc[:, 1], c=y.map({0: 'blue', 1: 'red'}), s=20, edgecolor='k')

plt.xlabel('Mean Radius')
plt.ylabel('Mean Concave Points')
plt.title('Decision Tree Decision Boundary')

# Ajuster la portée de l'axe y
plt.ylim(-0.01, 0.25)

plt.show()
