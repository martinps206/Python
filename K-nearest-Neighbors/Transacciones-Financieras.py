import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Cargar dataset
data = pd.read_csv('transactions.csv')

# Preprocesamiento de datos
features = ['Amount', 'Timestamp']
X = data[features]
y = data['Fraudulent']

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Entrenar modelo KNN para detección de anomalías
knn = LocalOutlierFactor(n_neighbors=20, novelty=True)
knn.fit(X_train[y_train == 0])  # Entrenar solo con transacciones no fraudulentas

# Predecir en el conjunto de prueba
y_pred = knn.predict(X_test)
y_pred = [1 if p == -1 else 0 for p in y_pred]  # Convertir a etiquetas binarias

# Evaluar el modelo
print(classification_report(y_test, y_pred))
