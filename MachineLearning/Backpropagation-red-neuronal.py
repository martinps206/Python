import numpy as np

# Función de activación sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivada de la sigmoide
def sigmoid_derivada(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Función de costo (MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Derivada de la función de costo (MSE)
def mse_derivada(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# Definir la estructura de la red neuronal (una capa oculta)
class RedNeuronalSimple:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializar pesos y biases aleatoriamente
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    # Propagación hacia adelante
    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = sigmoid(self.z1)  # Activación de la primera capa
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        y_pred = sigmoid(self.z2)  # Salida de la red
        return y_pred

    # Backpropagation
    def backward(self, X, y_true, y_pred):
        # Error en la salida
        error_salida = mse_derivada(y_true, y_pred) * sigmoid_derivada(self.z2)

        # Gradientes para los pesos entre capa oculta y salida
        dw2 = np.dot(self.a1.T, error_salida)
        db2 = np.sum(error_salida, axis=0, keepdims=True)

        # Error en la capa oculta (regla de la cadena)
        error_oculta = np.dot(error_salida, self.w2.T) * sigmoid_derivada(self.z1)

        # Gradientes para los pesos entre entrada y capa oculta
        dw1 = np.dot(X.T, error_oculta)
        db1 = np.sum(error_oculta, axis=0)

        # Actualización de los pesos y biases
        learning_rate = 0.01
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2

    # Entrenamiento
    def entrenar(self, X, y_true, epochs=10000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y_true, y_pred)
            if epoch % 1000 == 0:
                loss = mse(y_true, y_pred)
                print(f"Epoch {epoch}, Costo: {loss:.4f}")

# Datos de entrada (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_true = np.array([[0], [1], [1], [0]])

# Crear y entrenar la red
red = RedNeuronalSimple(input_size=2, hidden_size=2, output_size=1)
red.entrenar(X, y_true)
