# Algoritmo Posición Falsa para raíces, busca en intervalo [a, b]
import numpy as np

# INGRESO
fx = lambda x: x**3 + 4*x**2 - 10
a = 1
b = 2
tolera = 0.001

# PROCEDIMIENTO
tramo = abs(b - a)
fa = fx(a)
fb = fx(b)

while tramo > tolera:
    c = b - fb * (a - b) / (fa - fb)
    fc = fx(c)
    cambia = np.sign(fa) * np.sign(fc)
    
    if cambia > 0:
        tramo = abs(c - a)
        a = c
        fa = fc
    else:
        tramo = abs(b - c)
        b = c
        fb = fc

raiz = c

# SALIDA
print('Raíz:', raiz)
print('Error:', tramo)
