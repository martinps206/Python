import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generar datos de clasificación
X, y = make_classification(n_samples=200, n_features=10, n_informative=8, n_classes=2, random_state=42)
y = y.reshape(-1, 1)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inicializar los parámetros
theta = np.random.randn(X_train.shape[1], 1)
bias = 0
learning_rate = 0.01
lambda_reg = 0.1  # Regularización L1
epochs = 1000

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Predicción
def predict(X, theta, bias):
    return sigmoid(np.dot(X, theta) + bias)

# Costo con regularización L1 (Lasso)
def compute_cost(X, y, theta, bias, lambda_reg):
    m = X.shape[0]
    predictions = predict(X, theta, bias)
    cost = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    reg_term = (lambda_reg / m) * np.sum(np.abs(theta))
    return cost + reg_term

# Gradiente Descendente con Lasso
def gradient_descent(X, y, theta, bias, learning_rate, lambda_reg):
    m = X.shape[0]
    predictions = predict(X, theta, bias)
    
    d_theta = (1 / m) * np.dot(X.T, (predictions - y)) + (lambda_reg / m) * np.sign(theta)
    d_bias = (1 / m) * np.sum(predictions - y)
    
    # Actualizar parámetros
    theta -= learning_rate * d_theta
    bias -= learning_rate * d_bias
    
    return theta, bias

# Entrenamiento del modelo
costs = []
for epoch in range(epochs):
    theta, bias = gradient_descent(X_train, y_train, theta, bias, learning_rate, lambda_reg)
    cost = compute_cost(X_train, y_train, theta, bias, lambda_reg)
    costs.append(cost)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Costo: {cost:.4f}")

# Predicción en los datos de prueba
y_pred = (predict(X_test, theta, bias) >= 0.5).astype(int)

# Evaluación
accuracy = np.mean(y_pred == y_test) * 100
print(f"Exactitud del modelo: {accuracy:.2f}%")
