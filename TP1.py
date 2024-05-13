import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Charger les données à partir du fichier Excel
data = pd.read_csv("auto-mpg.csv")

# Afficher les premières lignes du jeu de données pour vérification
print(data.head())

# Diviser les données en fonction des caractéristiques (X) et de la cible (y)
X = data[['displacement']]
y = data['mpg']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création de l'objet de régression linéaire
regression = LinearRegression()

# Entraînement du modèle de régression linéaire sur l'ensemble d'entraînement
regression.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
predictions = regression.predict(X_test)

# Tracer les prédictions par rapport aux valeurs réelles
plt.scatter(X_test, y_test)
plt.plot(X_test, regression.predict(X_test))
plt.xlabel("Predictions")
plt.ylabel("Mileage")
plt.title("Mileage vs Predictions")
plt.show()

# Évaluation de la performance du modèle
from sklearn.metrics import mean_squared_error, r2_score

print("Mean squared error:", mean_squared_error(y_test, predictions))
print("Coefficient of determination (R^2):", r2_score(y_test, predictions))
