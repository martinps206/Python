# Algoritmo Posición Falsa para raíces
import numpy as np

# INGRESO
fx = lambda x: x**3 + 4*(x**2) - 10
a = 1
b = 2
tolera = 0.001

# PROCEDIMIENTO
tabla = []
tramo = abs(b - a)
fa = fx(a)
fb = fx(b)

while not (tramo <= tolera):
    c = b - fb * (a - b) / (fa - fb)
    fc = fx(c)
    tabla.append([a, c, b, fa, fc, fb, tramo])
    cambio = np.sign(fa) * np.sign(fc)
    
    if cambio > 0:
        tramo = abs(c - a)
        a = c
        fa = fc
    else:
        tramo = abs(b - c)
        b = c
        fb = fc

# Convierte la lista a un arreglo
tabla = np.array(tabla)

# SALIDA
np.set_printoptions(precision=4)
for i in range(len(tabla)):
    print('Iteracion:', i)
    print('[a, c, b]:', tabla[i, 0:3])
    print('[fa, fc, fb]:', tabla[i, 3:6])
    print('[tramo]:', tabla[i, 6])

print('Raiz:', c)
print('Error:', tramo)
