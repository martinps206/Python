import numpy as np
import os

# Ruta al .npz (usa ruta relativa si estás en la carpeta del proyecto)
f = os.path.abspath('fases_global_relativa_demo.npz')

d = np.load(f, allow_pickle=True)

print('Claves en el .npz:', list(d.keys()))

# Acceder a arrays por nombre
state = d['state']                # vectores flotantes complejos
state_with_global = d['state_with_global']
ket_plus = d['ket_plus']
ket_i = d['ket_i']

print('state =', state)
print('state_with_global =', state_with_global)
print('|+> =', ket_plus)
print('|i> =', ket_i)

# Ejemplo: calcular probabilidades en Z y X a partir de los datos guardados
def prob_in_computational(state):
    a, b = state
    return abs(a)**2, abs(b)**2

def plus_minus_basis():
    import numpy as np
    s2 = np.sqrt(2)
    plus = np.array([1, 1]) / s2
    minus = np.array([1, -1]) / s2
    return plus, minus

def prob_in_basis(state, basis_state):
    import numpy as np
    amp = np.vdot(basis_state, state)
    return abs(amp)**2

p0, p1 = prob_in_computational(state)
plus, minus = plus_minus_basis()
p_plus = prob_in_basis(state, plus)
p_minus = prob_in_basis(state, minus)

print('Prob Z:', p0, p1)
print('Prob X:', p_plus, p_minus)

d.close()