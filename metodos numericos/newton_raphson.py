# Método de Newton-Raphson
import numpy as np

# INGRESO
fx = lambda x: x**3 + 4*(x**2) - 10
dfx = lambda x: 3*(x**2) + 8*x
x0 = 2
tolera = 0.001

# PROCEDIMIENTO
tabla = []
tramo = abs(2 * tolera)
xi = x0
while tramo >= tolera:
    xnuevo = xi - fx(xi) / dfx(xi)
    tramo = abs(xnuevo - xi)
    tabla.append([xi, xnuevo, tramo])
    xi = xnuevo

# convierte la lista a un arreglo.
tabla = np.array(tabla)

# SALIDA
print(['xi', 'xnuevo', 'tramo'])
np.set_printoptions(precision=4)
print(tabla)
print('Raíz en: ', xi)
print('Con error de: ', tramo)
