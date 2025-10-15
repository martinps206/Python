import numpy as np

# Función para calcular MSE
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Función para calcular las derivadas parciales del MSE
def derivadas_mse(x, y_true, beta_0, beta_1):
    n = len(y_true)
    y_pred = beta_0 + beta_1 * x
    
    # Derivada con respecto a beta_0 (intercepto)
    d_beta_0 = -2 * np.mean(y_true - y_pred)
    
    # Derivada con respecto a beta_1 (pendiente)
    d_beta_1 = -2 * np.mean((y_true - y_pred) * x)
    
    return d_beta_0, d_beta_1

# Datos de ejemplo (valores simulados)
x = np.array([1, 2, 3, 4, 5])
y_true = np.array([2, 3, 5, 7, 11])  # Estos son valores reales

# Valores iniciales para los parámetros
beta_0 = 0.0  # intercepto inicial
beta_1 = 0.0  # pendiente inicial

# Número de iteraciones para el descenso de gradiente
epochs = 1000
learning_rate = 0.01

# Descenso de gradiente
for epoch in range(epochs):
    y_pred = beta_0 + beta_1 * x  # Predicción del modelo
    
    # Cálculo del error y las derivadas
    mse_val = mse(y_true, y_pred)
    d_beta_0, d_beta_1 = derivadas_mse(x, y_true, beta_0, beta_1)
    
    # Actualización de los parámetros
    beta_0 -= learning_rate * d_beta_0
    beta_1 -= learning_rate * d_beta_1
    
    # Mostrar el progreso cada 100 iteraciones
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: MSE = {mse_val:.4f}, beta_0 = {beta_0:.4f}, beta_1 = {beta_1:.4f}")

print(f"\nModelo final: y = {beta_0:.4f} + {beta_1:.4f} * x")
