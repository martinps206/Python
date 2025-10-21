# Aplicación demostrativa: Fases globales y relativas (nivel matemático avanzado)
# Esta celda ejecuta un programa Python que:
# 1. Traduce simbólicamente por qué una fase global no afecta probabilidades.
# 2. Calcula numéricamente probabilidades en base Z y base X para un estado con y sin fase global.
# 3. Muestra un ejemplo con fase relativa que *sí* cambia resultados en la base X.
# 4. Grafica la esfera de Bloch y ubica los estados relevantes (puntos de la esfera).
# 5. Presenta resultados numéricos y simplificaciones simbólicas.
#
# Requisitos: numpy, sympy, matplotlib
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import os

# --- Funciones matemáticas ---
def normalize(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("Estado de norma cero")
    return state / norm

def apply_global_phase(state: np.ndarray, theta: float) -> np.ndarray:
    return np.exp(1j * theta) * state

def prob_in_computational(state: np.ndarray):
    # state is 2-vector [a, b]
    a, b = state
    return np.abs(a)**2, np.abs(b)**2

def plus_minus_basis():
    s2 = np.sqrt(2)
    plus = np.array([1, 1]) / s2
    minus = np.array([1, -1]) / s2
    return plus, minus

def prob_in_basis(state: np.ndarray, basis_state: np.ndarray):
    amp = np.vdot(basis_state, state)  # inner product <basis|state>
    return np.abs(amp)**2

# Bloch vector for a pure qubit |ψ> = [a, b]^T
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def bloch_vector(state: np.ndarray):
    state = state.reshape(2,1)
    rho = state @ state.conj().T  # projector |ψ><ψ|
    x = np.real(np.trace(rho @ sigma_x))
    y = np.real(np.trace(rho @ sigma_y))
    z = np.real(np.trace(rho @ sigma_z))
    return np.array([x, y, z])

# --- Ejemplo concreto (el mostrado en la imagen) ---
# Estado: (sqrt(3)/2) |0> + (1/2) |1>
a = np.sqrt(3)/2
b = 1/2
state = normalize(np.array([a, b], dtype=complex))

# Aplicar fase global e.g. theta = 1.23 rad
theta = 1.23
state_with_global = apply_global_phase(state, theta)

# Probabilidades en base Z sin y con fase global
p0, p1 = prob_in_computational(state)
p0_g, p1_g = prob_in_computational(state_with_global)

# Probabilidades en base X (|+>, |->)
plus, minus = plus_minus_basis()
p_plus = prob_in_basis(state, plus)
p_minus = prob_in_basis(state, minus)
p_plus_g = prob_in_basis(state_with_global, plus)
p_minus_g = prob_in_basis(state_with_global, minus)

# --- Ejemplo de fase relativa ---
# |+> = (|0> + |1>) / sqrt(2)
ket_plus = plus
# |i> = (|0> + i |1>) / sqrt(2)
ket_i = normalize(np.array([1, 1j]) / np.sqrt(2))

# Probabilidades Z para |+> y |i>
pz_plus = prob_in_computational(ket_plus)
pz_i = prob_in_computational(ket_i)

# Probabilidades X para |+> y |i>
px_plus = (prob_in_basis(ket_plus, plus), prob_in_basis(ket_plus, minus))
px_i = (prob_in_basis(ket_i, plus), prob_in_basis(ket_i, minus))

# --- Simbólico: demostrar que fase global no afecta probabilidades ---
theta_s = sp.symbols('theta', real=True)
a_s, b_s = sp.symbols('a b', complex=True)
exp_i_theta = sp.exp(sp.I * theta_s)
# Probabilidades simbólicas |e^{iθ} a|^2 = a * conj(a) (el factor e^{iθ} cancela)
prob_a_symbolic = sp.simplify(sp.Abs(exp_i_theta * a_s)**2)
prob_b_symbolic = sp.simplify(sp.Abs(exp_i_theta * b_s)**2)

# --- Resultados Numéricos y Salidas ---
print("Estado base (normalizado): a = {:.6f}, b = {:.6f}".format(state[0], state[1]))
print(f"Probabilidades en Z sin fase global: P(|0>)={p0:.6f}, P(|1>)={p1:.6f}")
print(f"Probabilidades en Z con fase global θ={theta:.3f}: P(|0>)={p0_g:.6f}, P(|1>)={p1_g:.6f}")
print("-> Igualdad numérica confirma que la fase global no cambia probabilidades en la base Z.\n")

print(f"Probabilidades en X sin fase global: P(+)={p_plus:.6f}, P(-)={p_minus:.6f}")
print(f"Probabilidades en X con fase global θ={theta:.3f}: P(+)={p_plus_g:.6f}, P(-)={p_minus_g:.6f}")
print("-> Igualdad numérica confirma que la fase global no cambia probabilidades en la base X tampoco.\n")

print("Ejemplo fase relativa:")
print(f"|+> en Z: P(|0>)={pz_plus[0]:.6f}, P(|1>)={pz_plus[1]:.6f}")
print(f"|i> en Z: P(|0>)={pz_i[0]:.6f}, P(|1>)={pz_i[1]:.6f}")
print("-> En la base Z las estadisticas coinciden (ambos dan 1/2, 1/2).\n")

print("Mediciones en la base X:")
print(f"Medir |+> en X: P(+)={px_plus[0]:.6f}, P(-)={px_plus[1]:.6f}")
print(f"Medir |i> en X: P(+)={px_i[0]:.6f}, P(-)={px_i[1]:.6f}")
print("-> Aquí se ve la diferencia: |+> siempre da + en X; |i> da ± con probabilidad 1/2.\n")

print("Simbólico: |e^{iθ} a|^2 simplifica a: ", prob_a_symbolic)
print("Simbólico: |e^{iθ} b|^2 simplifica a: ", prob_b_symbolic)

# --- Graficar esfera de Bloch y puntos relevantes ---
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
# Esfera
u = np.linspace(0, 2 * np.pi, 80)
v = np.linspace(0, np.pi, 40)
x_s = np.outer(np.cos(u), np.sin(v))
y_s = np.outer(np.sin(u), np.sin(v))
z_s = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(x_s, y_s, z_s, alpha=0.12, linewidth=0)  # no especificamos colores
# Puntos: estado dado, estado con fase global, |+>, |i>
bv_state = bloch_vector(state)
bv_state_g = bloch_vector(state_with_global)
bv_plus = bloch_vector(ket_plus)
bv_i = bloch_vector(ket_i)

ax.scatter([bv_state[0]], [bv_state[1]], [bv_state[2]], s=80)
ax.text(bv_state[0], bv_state[1], bv_state[2], " ψ (sin fase)", size=10)
ax.scatter([bv_state_g[0]], [bv_state_g[1]], [bv_state_g[2]], s=80)
ax.text(bv_state_g[0], bv_state_g[1], bv_state_g[2], " ψ (con fase global)", size=10)
ax.scatter([bv_plus[0]], [bv_plus[1]], [bv_plus[2]], s=80)
ax.text(bv_plus[0], bv_plus[1], bv_plus[2], " |+>", size=10)
ax.scatter([bv_i[0]], [bv_i[1]], [bv_i[2]], s=80)
ax.text(bv_i[0], bv_i[1], bv_i[2], " |i>", size=10)

# Ejes
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Esfera de Bloch: puntos relevantes (global vs relativa)')
# Ajuste de escala
ax.set_box_aspect([1,1,1])
plt.show()

# --- Final: empaquetar funciones para uso interactivo / reusabilidad ---
def demonstrate_state(a, b, theta=0.0):
    st = normalize(np.array([a, b], dtype=complex))
    stg = apply_global_phase(st, theta)
    results = {
        "state": st,
        "state_with_global": stg,
        "prob_Z": prob_in_computational(st),
        "prob_Z_global": prob_in_computational(stg),
        "prob_X": (prob_in_basis(st, plus), prob_in_basis(st, minus)),
        "prob_X_global": (prob_in_basis(stg, plus), prob_in_basis(stg, minus)),
        "bloch_state": bloch_vector(st),
        "bloch_state_global": bloch_vector(stg)
    }
    return results

# Guardar pequeño resumen numérico en un .npz para posible descarga posterior
try:
    out_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fases_global_relativa_demo.npz'))
except NameError:
    # __file__ may not be defined in some interactive environments; fall back to cwd
    out_filename = os.path.abspath('fases_global_relativa_demo.npz')

try:
    np.savez(out_filename,
             state=state, state_with_global=state_with_global,
             ket_plus=ket_plus, ket_i=ket_i)
    print(f"\nArchivo de resultados guardado en: {out_filename}")
except Exception as e:
    print("\nNo se pudo guardar el archivo .npz:", e)
