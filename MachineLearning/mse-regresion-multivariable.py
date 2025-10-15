import numpy as np

# Función para calcular MSE en el caso multivariable
def mse_multivariable(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Función para calcular las derivadas parciales del MSE multivariable
def derivadas_mse_multivariable(X, y_true, betas):
    n = len(y_true)
    y_pred = np.dot(X, betas)  # Producto punto entre los datos de entrada y los coeficientes
    
    # Derivadas parciales con respecto a cada beta
    d_betas = -2 * np.dot(X.T, (y_true - y_pred)) / n
    
    return d_betas

# Datos de ejemplo con múltiples variables
X = np.array([
    [1, 1, 1],
    [1, 2, 4],
    [1, 3, 9],
    [1, 4, 16],
    [1, 5, 25]
])
y_true = np.array([2, 3, 5, 7, 11])

# Inicializar los parámetros beta (incluye el intercepto como el primer valor)
betas = np.zeros(X.shape[1])

# Parámetros del descenso de gradiente
epochs = 1000
learning_rate = 0.01

# Descenso de gradiente
for epoch in range(epochs):
    y_pred = np.dot(X, betas)  # Predicción del modelo
    mse_val = mse_multivariable(y_true, y_pred)
    
    # Calcular las derivadas
    d_betas = derivadas_mse_multivariable(X, y_true, betas)
    
    # Actualizar los parámetros
    betas -= learning_rate * d_betas
    
    # Mostrar el progreso cada 100 iteraciones
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: MSE = {mse_val:.4f}, Betas = {betas}")

print(f"\nModelo final: Betas = {betas}")
