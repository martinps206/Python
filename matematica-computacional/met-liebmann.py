import numpy as np

# Parámetros
L = 4  # Longitud de la placa
h = 1  # Paso en la cuadrícula
omega = 1.5  # Factor de sobrerelajación
es = 1.0  # Error relativo esperado
max_iter = 1000  # Máximo número de iteraciones

# Inicializar la cuadrícula
n = int(L/h) + 1  # Número de nodos por lado
u = np.zeros((n, n))

# Condiciones de frontera
u[:, 0] = 20  # Frontera inferior
u[:, -1] = 160  # Frontera superior
u[0, :] = 80  # Frontera izquierda
u[-1, :] = 0  # Frontera derecha

# Iteraciones
for k in range(max_iter):
    u_old = u.copy()
    max_error = 0
    for i in range(1, n-1):
        for j in range(1, n-1):
            u_new = (1 - omega) * u[i, j] + (omega / 4) * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
            error = abs((u_new - u[i, j]) / u_new) * 100
            max_error = max(max_error, error)
            u[i, j] = u_new

    if max_error < es:
        break

# Mostrar el resultado
print("Temperatura de la placa:")
print(u)
print(f"Iteraciones realizadas: {k+1}")
print(f"Error máximo alcanzado: {max_error}%")
