import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generar datos de muestra
X, y = make_regression(n_samples=100, n_features=1, noise=10)
y = y.reshape(-1, 1)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inicializar los parámetros
theta = np.random.randn(X_train.shape[1], 1)
bias = 0
learning_rate = 0.01
lambda_reg = 0.1  # Hiperparámetro de regularización L2
epochs = 1000

# Definir las funciones de activación y costo regularizado (L2)
def predict(X, theta, bias):
    return np.dot(X, theta) + bias

def compute_cost(X, y, theta, bias, lambda_reg):
    m = X.shape[0]
    predictions = predict(X, theta, bias)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    reg_term = (lambda_reg / (2 * m)) * np.sum(theta ** 2)
    return cost + reg_term

def gradient_descent(X, y, theta, bias, learning_rate, lambda_reg):
    m = X.shape[0]
    predictions = predict(X, theta, bias)
    
    # Gradientes
    d_theta = (1 / m) * np.dot(X.T, (predictions - y)) + (lambda_reg / m) * theta
    d_bias = (1 / m) * np.sum(predictions - y)
    
    # Actualizar los parámetros
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

# Predicción y evaluación
y_pred = predict(X_test, theta, bias)

# Visualizar los resultados
plt.scatter(X_test, y_test, color="blue", label="Datos reales")
plt.plot(X_test, y_pred, color="red", label="Predicción")
plt.title("Regresión Lineal con Regularización L2")
plt.xlabel("Características")
plt.ylabel("Precio")
plt.legend()
plt.show()
