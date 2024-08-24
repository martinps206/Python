import numpy as np
import scipy.linalg as la

# Parámetros
L = 8.0  # Longitud del alambre en cm
T = 0.5  # Tiempo final en segundos
h = 2.0  # Paso espacial en cm
dt = 0.1  # Paso temporal en s
alpha = 108.56  # Difusividad térmica en cm^2/s

# Discretización
N_x = int(L / h) + 1  # Número de puntos espaciales
N_t = int(T / dt) + 1  # Número de puntos temporales
r = alpha * dt / (h**2)  # Factor del método de Crank-Nicholson

# Condiciones iniciales
x = np.linspace(0, L, N_x)
u = np.exp(x**2 + 8*x)  # Distribución inicial
u[0] = 1.0  # Condición de frontera en x=0
u[-1] = 1.0  # Condición de frontera en x=L

# Matrices del sistema
A = np.zeros((N_x-2, N_x-2))
B = np.zeros((N_x-2, N_x-2))

# Llenar matrices A y B
for i in range(1, N_x-1):
    if i == 1:
        A[i-1,i-1] = 1 + r
        A[i-1,i] = -r/2
        B[i-1,i-1] = 1 - r
        B[i-1,i] = r/2
    elif i == N_x-2:
        A[i-1,i-1] = 1 + r
        A[i-1,i-2] = -r/2
        B[i-1,i-1] = 1 - r
        B[i-1,i-2] = r/2
    else:
        A[i-1,i-1] = 1 + r
        A[i-1,i-2] = -r/2
        A[i-1,i] = -r/2
        B[i-1,i-1] = 1 - r
        B[i-1,i-2] = r/2
        B[i-1,i] = r/2

# Solución temporal
u_sol = np.zeros((N_t, N_x))
u_sol[0, :] = u

# Iteración temporal
for n in range(1, N_t):
    b = B @ u[1:-1]  # Vector de términos independientes
    b[0] += r/2 * (u[0] + u_sol[n-1,0])
    b[-1] += r/2 * (u[-1] + u_sol[n-1,-1])
    u_new = la.solve(A, b)
    u_sol[n, 1:-1] = u_new
    u_sol[n, 0] = 1.0  # Condición de frontera en x=0
    u_sol[n, -1] = 1.0  # Condición de frontera en x=L

# Mostrar resultados para t=0.5s
print(f"Distribución de temperatura en t={T} segundos:")
print(u_sol[-1, :])
