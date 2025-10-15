import numpy as np

# Función de activación Sigmoide
def sigmoid(z):
    """
    Calcula la función sigmoide.

    Parameters:
    z (numpy.ndarray): Entrada a la función sigmoide.

    Returns:
    numpy.ndarray: Salida después de aplicar la función sigmoide.
    """
    return 1 / (1 + np.exp(-z))

# Derivada de la función Sigmoide
def sigmoid_derivada(z):
    """
    Calcula la derivada de la función sigmoide.

    Parameters:
    z (numpy.ndarray): Entrada a la función sigmoide.

    Returns:
    numpy.ndarray: Derivada de la función sigmoide.
    """
    s = sigmoid(z)
    return s * (1 - s)

# Función de activación ReLU
def relu(z):
    """
    Calcula la función de activación ReLU.

    Parameters:
    z (numpy.ndarray): Entrada a la función ReLU.

    Returns:
    numpy.ndarray: Salida después de aplicar ReLU.
    """
    return np.maximum(0, z)

# Derivada de la función ReLU
def relu_derivada(z):
    """
    Calcula la derivada de la función ReLU.

    Parameters:
    z (numpy.ndarray): Entrada a la función ReLU.

    Returns:
    numpy.ndarray: Derivada de ReLU.
    """
    return np.where(z > 0, 1, 0)

# Función de costo MSE
def mse(y_true, y_pred):
    """
    Calcula el Error Cuadrático Medio (MSE).

    Parameters:
    y_true (numpy.ndarray): Valores reales.
    y_pred (numpy.ndarray): Valores predichos por el modelo.

    Returns:
    float: Valor del MSE.
    """
    return np.mean((y_true - y_pred) ** 2)

# Derivada de la función de costo MSE
def mse_derivada(y_true, y_pred):
    """
    Calcula la derivada del MSE con respecto a las predicciones.

    Parameters:
    y_true (numpy.ndarray): Valores reales.
    y_pred (numpy.ndarray): Valores predichos por el modelo.

    Returns:
    numpy.ndarray: Derivada del MSE.
    """
    return 2 * (y_pred - y_true) / y_true.size

# Clase para la Red Neuronal Profunda
class RedNeuronalProfunda:
    def __init__(self, estructura):
        """
        Inicializa la red neuronal con la estructura especificada.

        Parameters:
        estructura (list): Lista que contiene el número de neuronas en cada capa.
                           Por ejemplo, [2, 3, 2, 1] representa una red con:
                           - 2 neuronas de entrada
                           - 2 capas ocultas con 3 y 2 neuronas respectivamente
                           - 1 neurona de salida
        """
        self.capas = len(estructura) - 1  # Número de capas de pesos
        self.pesos = []
        self.biases = []
        
        # Inicialización de pesos y biases
        for i in range(self.capas):
            # Inicialización con distribución normal estándar
            self.pesos.append(np.random.randn(estructura[i], estructura[i + 1]) * np.sqrt(2 / estructura[i]))
            self.biases.append(np.zeros((1, estructura[i + 1])))
    
    # Propagación hacia adelante
    def forward(self, X):
        """
        Realiza la propagación hacia adelante.

        Parameters:
        X (numpy.ndarray): Datos de entrada.

        Returns:
        numpy.ndarray: Salida de la red neuronal.
        """
        self.activaciones = [X]
        self.z = []
        
        for i in range(self.capas):
            z = np.dot(self.activaciones[-1], self.pesos[i]) + self.biases[i]
            self.z.append(z)
            if i == self.capas - 1:  # Última capa (output)
                a = sigmoid(z)
            else:
                a = relu(z)
            self.activaciones.append(a)
        
        return self.activaciones[-1]
    
    # Backpropagation
    def backward(self, X, y_true, y_pred):
        """
        Realiza el algoritmo de backpropagation para actualizar pesos y biases.

        Parameters:
        X (numpy.ndarray): Datos de entrada.
        y_true (numpy.ndarray): Valores reales.
        y_pred (numpy.ndarray): Valores predichos por la red.
        """
        # Calcular el error en la capa de salida
        error = mse_derivada(y_true, y_pred) * sigmoid_derivada(self.z[-1])
        
        # Gradientes para la última capa
        grad_w = np.dot(self.activaciones[-2].T, error)
        grad_b = np.sum(error, axis=0, keepdims=True)
        
        # Almacenar los gradientes
        gradientes_w = [grad_w]
        gradientes_b = [grad_b]
        
        # Propagación del error hacia atrás a través de las capas ocultas
        for i in range(self.capas - 2, -1, -1):
            error = np.dot(error, self.pesos[i + 1].T) * relu_derivada(self.z[i])
            grad_w = np.dot(self.activaciones[i].T, error)
            grad_b = np.sum(error, axis=0, keepdims=True)
            gradientes_w.insert(0, grad_w)
            gradientes_b.insert(0, grad_b)
        
        # Actualizar los pesos y biases con descenso de gradiente
        learning_rate = 0.01
        for i in range(self.capas):
            self.pesos[i] -= learning_rate * gradientes_w[i]
            self.biases[i] -= learning_rate * gradientes_b[i]
    
    # Entrenamiento de la red
    def entrenar(self, X, y_true, epochs=10000):
        """
        Entrena la red neuronal utilizando backpropagation.

        Parameters:
        X (numpy.ndarray): Datos de entrada.
        y_true (numpy.ndarray): Valores reales.
        epochs (int): Número de iteraciones de entrenamiento.
        """
        for epoch in range(epochs):
            # Propagación hacia adelante
            y_pred = self.forward(X)
            
            # Backpropagation para actualizar pesos y biases
            self.backward(X, y_true, y_pred)
            
            # Imprimir el costo cada 1000 iteraciones
            if epoch % 1000 == 0:
                loss = mse(y_true, y_pred)
                print(f"Epoch {epoch}, Costo: {loss:.4f}")
    
    # Predicción
    def predecir(self, X):
        """
        Realiza predicciones con la red neuronal entrenada.

        Parameters:
        X (numpy.ndarray): Datos de entrada.

        Returns:
        numpy.ndarray: Predicciones de la red neuronal.
        """
        y_pred = self.forward(X)
        return y_pred

# Función para redondear las predicciones a 0 o 1
def redondear_predicciones(y_pred):
    """
    Redondea las predicciones continuas a valores binarios (0 o 1).

    Parameters:
    y_pred (numpy.ndarray): Predicciones continuas de la red neuronal.

    Returns:
    numpy.ndarray: Predicciones redondeadas.
    """
    return np.where(y_pred >= 0.5, 1, 0)

# Configuración y entrenamiento de la red neuronal
if __name__ == "__main__":
    # Definir la estructura de la red: 2 entradas, dos capas ocultas (3 y 2 neuronas), 1 salida
    estructura = [2, 3, 2, 1]
    
    # Datos de entrada (XOR)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_true = np.array([
        [0],
        [1],
        [1],
        [0]
    ])
    
    # Crear la red neuronal profunda
    red_profundidad = RedNeuronalProfunda(estructura)
    
    # Entrenar la red
    red_profundidad.entrenar(X, y_true, epochs=10000)
    
    # Realizar predicciones
    y_pred = red_profundidad.predecir(X)
    y_pred_redondeado = redondear_predicciones(y_pred)
    
    # Mostrar los resultados
    print("\nResultados de las predicciones:")
    for i in range(len(X)):
        print(f"Entrada: {X[i]} -> Predicción: {y_pred_redondeado[i][0]} (Valor continuo: {y_pred[i][0]:.4f})")
