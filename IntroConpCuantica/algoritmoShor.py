# Implementación didáctica de Shor (simulación de la parte cuántica) 
# - Propósito: demostrar el subrutina de búsqueda de periodo con QFT simulada con vectores de estado.
# - Limitaciones: se simula exactamente el estado cuántico (vector de amplitudes). Funciona para N pequeños (p.ej. 15, 21, 33).
# - Resultado: intenta recuperar el periodo r para luego factorizar N (si r es par y cumple condiciones).

import numpy as np
from math import gcd, log2, ceil
from fractions import Fraction
import random

# Utilidades matemáticas
def continued_fraction_expansion(x, max_denominator):
    """Aproxima x mediante fracción con denominador <= max_denominator usando continued fractions (Fraction)."""
    return Fraction(x).limit_denominator(max_denominator)

def get_factors_from_r(a, r, N):
    """Dadas a, r intentamos obtener factores de N mediante gcd(a^(r/2) +- 1, N)."""
    if r % 2 != 0:
        return None
    ar2 = pow(a, r//2, N)
    if ar2 == N-1:  # a^(r/2) ≡ -1 (mod N)
        return None
    p = gcd(ar2 - 1, N)
    q = gcd(ar2 + 1, N)
    if 1 < p < N:
        return p, N//p
    if 1 < q < N:
        return q, N//q
    return None

# QFT sobre t qubits (matriz 2^t x 2^t)
def qft_matrix(t):
    dim = 1 << t
    omega = np.exp(2j * np.pi / dim)
    j = np.arange(dim)
    k = j.reshape((dim,1))
    M = omega ** (k * j) / np.sqrt(dim)
    return M

# Inversa de QFT
def iqft_matrix(t):
    return np.conjugate(qft_matrix(t)).T

def simulate_shor(N, a=None, t=None, shots=1024, seed=None):
    """
    Simula la subrutina de búsqueda de periodo de Shor para un N compuesto pequeño.
    - a: base aleatoria coprima con N (si no se da, se elige aleatoriamente)
    - t: número de qubits del registro de fase (si no se da, se usa 2*ceil(log2(N)))
    Devuelve estimaciones del periodo r y un intento de factorización.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # Elegir 'a' coprimo con N
    if a is None:
        while True:
            a = random.randrange(2, N)
            if gcd(a, N) == 1:
                break
    if gcd(a, N) != 1:
        return {"error": f"a={a} no es coprimo con N; gcd={gcd(a,N)}", "a": a}
    
    # t: número de qubits del registro de fase (precision). Recomendado: 2 * ceil(log2(N))
    if t is None:
        t = 2 * ceil(log2(N))
    dim_phase = 1 << t
    m = ceil(log2(N))  # qubits del registro del resultado modular (segunda parte)
    dim_target = 1 << m
    # Estado inicial: |0>^{t} \otimes |1> (target en valor 1)
    # Tras Hadamards en registro fase -> superposición |x> para x in [0, 2^t-1]
    # Aplicamos la exponenciación modular controlada: |x>|1> -> |x>|a^x mod N>
    # Finalmente aplicamos IQFT en el registro fase y medimos.
    
    # Construimos el vector de amplitudes para el sistema combinado (fase x target)
    total_dim = dim_phase * dim_target
    state = np.zeros(total_dim, dtype=complex)
    # Inicialmente, after H on phase: amplitude for |x> is 1/sqrt(dim_phase), target is |1>
    for x in range(dim_phase):
        y = pow(a, x, N)  # a^x mod N
        # target register in computational basis indexes values 0..(dim_target-1), we use value y
        idx = x * dim_target + y
        state[idx] = 1/np.sqrt(dim_phase)
    
    # Reinterpret as matrix with shape (dim_phase, dim_target) for convenience
    state_matrix = state.reshape((dim_phase, dim_target))
    # After modular exponentiation we have state_matrix[x, y] nonzero only at y = a^x mod N
    
    # Now apply inverse QFT on the phase register.
    iqft = iqft_matrix(t)
    # Applying IQFT to the phase register: for each target basis state apply iqft column-wise
    # Compute new_phase_state = (IQFT ⊗ I_target) * state_vector
    # Efficiently: new_matrix = iqft @ state_matrix
    new_matrix = iqft @ state_matrix  # shape (dim_phase, dim_target)
    new_state = new_matrix.reshape(total_dim)
    
    # Now simulate measurements on the phase register only: probabilities summing over target register
    probs_phase = np.zeros(dim_phase)
    for x in range(dim_phase):
        # Probability of measuring 'x' on phase register is sum over target of |amplitude|^2
        probs_phase[x] = np.sum(np.abs(new_matrix[x, :])**2)
    # Sample measurements (shots)
    samples = np.random.choice(np.arange(dim_phase), size=shots, p=probs_phase)
    # Count occurrences (histogram)
    unique, counts = np.unique(samples, return_counts=True)
    counts_dict = dict(zip(unique.tolist(), counts.tolist()))
    
    # For each frequent measured value, attempt to recover r via continued fractions
    estimates = []
    sorted_counts = sorted(counts_dict.items(), key=lambda kv: kv[1], reverse=True)
    # Try top K outcomes
    K = min(8, len(sorted_counts))
    for i in range(K):
        measured, cnt = sorted_counts[i]
        phase = measured / dim_phase
        frac = continued_fraction_expansion(phase, max_denominator=N)  # denominator upto N
        r_candidate = frac.denominator
        # It's possible that continued fraction gives small denominator; try to refine (sometimes multiple)
        # Check if a^r_candidate mod N == 1; if not consider multiples
        found_r = None
        for mult in range(1, 6):  # try small multiples
            r_try = r_candidate * mult
            if pow(a, r_try, N) == 1:
                found_r = r_try
                break
        estimates.append({
            "measured": measured,
            "counts": cnt,
            "phase": phase,
            "fraction": f"{frac.numerator}/{frac.denominator}",
            "r_candidate": r_candidate,
            "r_found": found_r
        })
    
    # Try to extract factors from any found r
    factors = None
    for est in estimates:
        if est["r_found"] is None:
            continue
        r = est["r_found"]
        f = get_factors_from_r(a, r, N)
        if f is not None:
            factors = {"a": a, "r": r, "factors": f}
            break
    
    result = {
        "N": N,
        "a": a,
        "t": t,
        "dim_phase": dim_phase,
        "dim_target": dim_target,
        "counts": counts_dict,
        "estimates": estimates,
        "factors": factors
    }
    return result

# Prueba demostrativa con N=15 (caso clásico de ejemplo) y otra prueba con N=21
for N in (15, 21):
    print(f"==== Simulación Shor para N = {N} ====")
    # Intentamos hasta cierto número de 'a' para tener una buena demostración
    attempts = 0
    success = None
    while attempts < 6 and success is None:
        a_try = random.randrange(2, N)
        if gcd(a_try, N) != 1:
            print(f"  a={a_try} comparte factor triv. => gcd={gcd(a_try,N)} (factor encontrado)")
            success = {"N": N, "a": a_try, "factors": (gcd(a_try,N), N//gcd(a_try,N))}
            break
        res = simulate_shor(N, a=a_try, t=2*ceil(log2(N)), shots=2048, seed=42+attempts)
        print(f"  Intento {attempts+1}: a={a_try}, counts_top3 = {sorted(res['counts'].items(), key=lambda kv: kv[1], reverse=True)[:3]}")
        if res["factors"]:
            print(f"  => Factores encontrados: {res['factors']} (r={res['factors']['r']})")
            success = res
            break
        attempts += 1
    if success is None:
        print("  No se encontraron factores con los intentos realizados (intentar más runs o diferentes a).")
    else:
        print("  Resultado (resumen):", success)
    print("")

