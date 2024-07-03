import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos
g = 9.81  # Aceleración debida a la gravedad (m/s^2)
sigma = 0.074  # Tensión superficial del agua (N/m)
rho = 1000  # Densidad del agua (kg/m^3)

# Definir la relación de dispersión
def dispersion_relation(k):
    return np.sqrt(g * k + (sigma * k**3) / rho)

# Espacio y tiempo
L = 10  # Longitud del dominio (m)
N = 1000  # Número de puntos en el espacio
x = np.linspace(0, L, N)
dx = x[1] - x[0]

T = 2  # Tiempo total de simulación (s)
dt = 0.01  # Paso de tiempo (s)
t = np.arange(0, T, dt)

# Condición inicial: una onda senoidal
A = 1.0  # Amplitud de la onda inicial
k0 = 2 * np.pi / L  # Número de onda de la onda inicial
y0 = A * np.sin(k0 * x)

# Transformada de Fourier de la condición inicial
y0_hat = np.fft.fft(y0)

# Evolución temporal usando la relación de dispersión
y_t = np.zeros((len(t), len(x)), dtype=complex)
for i, ti in enumerate(t):
    omega = dispersion_relation(k0)
    y_t[i, :] = np.fft.ifft(y0_hat * np.exp(-1j * omega * ti))

# Graficar la evolución de la onda
fig, ax = plt.subplots()
for i in range(0, len(t), 10):
    ax.plot(x, np.real(y_t[i, :]), label=f't={t[i]:.2f}s')
ax.set_xlabel('Posición (m)')
ax.set_ylabel('Desplazamiento (m)')
ax.set_title('Evolución de la Onda')
plt.legend()
plt.show()
