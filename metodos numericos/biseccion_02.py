# Algoritmo de Bisección
# [a, b] se escogen de la gráfica de la función
# error = tolera
import numpy as np
import matplotlib.pyplot as plt

# INGRESO
fx = lambda x: x**3 + 4*x**2 - 10
a = 1
b = 2
tolera = 0.001

# PROCEDIMIENTO
tramo = b - a

while tramo > tolera:
    c = (a + b) / 2
    fa = fx(a)
    fb = fx(b)
    fc = fx(c)
    cambia = np.sign(fa) * np.sign(fc)
    
    if cambia < 0:
        b = c
    elif cambia > 0:
        a = c
    else:
        # Si fc es exactamente cero, encontramos la raíz exacta
        break

    tramo = b - a

# SALIDA
raiz = c
print('Raíz en:', raiz)
print('Error en tramo:', tramo)

# GRÁFICA
x_vals = np.linspace(a, b, 400)
y_vals = fx(x_vals)

plt.plot(x_vals, y_vals, label='f(x)')
plt.axhline(0, color="black")
plt.axvline(raiz, color="red", linestyle="--", label=f'Raíz: {raiz:.4f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Algoritmo de Bisección')
plt.grid()
plt.legend()
plt.show()
