# Algoritmo de Bisección con gráfica
import numpy as np
import matplotlib.pyplot as plt

# INGRESO
fx = lambda x: x**3 + 4*x**2 - 10
a = 1
b = 2
tolera = 0.001

# PROCEDIMIENTO
tabla = []
tramo = b - a
fa = fx(a)
fb = fx(b)
i = 1

while tramo > tolera:
    c = (a + b) / 2
    fc = fx(c)
    tabla.append([i, a, c, b, fa, fc, fb, tramo])
    i += 1

    cambia = np.sign(fa) * np.sign(fc)
    if cambia < 0:
        b = c
        fb = fc
    else:
        a = c
        fa = fc
    tramo = b - a

c = (a + b) / 2
fc = fx(c)
tabla.append([i, a, c, b, fa, fc, fb, tramo])
tabla = np.array(tabla)

raiz = c

# SALIDA
np.set_printoptions(precision=4)
print('[ i, a, c, b, f(a), f(c), f(b), tramo]')

# Tabla con formato
for fila in tabla:
    formato = '{:.0f} ' + ' '.join(['{:.3f}'] * (len(fila) - 1))
    print(formato.format(*fila))

print('Raíz:', raiz)

# GRÁFICA
xi = tabla[:, 2]
yi = tabla[:, 5]

# Ordena los puntos para la gráfica
orden = np.argsort(xi)
xi = xi[orden]
yi = yi[orden]

plt.plot(xi, yi, label='f(c)')
plt.plot(xi, yi, 'o', label='Puntos de c')
plt.axhline(0, color="black")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bisección en f(x)')
plt.grid()
plt.legend()
plt.show()
