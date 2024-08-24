import numpy as np

# Parámetros
h = 0.2
k = 0.05
r = 1
x_max = 1.0
t_max = 0.2
alpha = r * k / h**2

# Discretización del espacio y tiempo
M = int(x_max / h)
N = int(t_max / k)

# Matriz para almacenar soluciones
u = np.zeros((N+1, M+1))

# Condiciones iniciales (ejemplo: u(x, 0) = sin(pi * x))
x = np.linspace(0, x_max, M+1)
u[0, :] = np.sin(np.pi * x)

# Iteración usando diferencias finitas
for n in range(0, N):
    for i in range(1, M):
        u[n+1, i] = u[n, i] + alpha * (u[n, i+1] - 2*u[n, i] + u[n, i-1])

# Imprimir las primeras cuatro filas
print("Solución aproximada (primeras 4 filas):")
print(u[:4, :])
