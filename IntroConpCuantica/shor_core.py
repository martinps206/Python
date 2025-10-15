# shor_core.py
from math import gcd, ceil, log2
from fractions import Fraction
import numpy as np
import random

def continued_fraction_expansion(x, max_denominator):
    return Fraction(x).limit_denominator(max_denominator)

def get_factors_from_r(a, r, N):
    if r % 2 != 0:
        return None
    ar2 = pow(a, r//2, N)
    if ar2 == N-1:
        return None
    p = gcd(ar2 - 1, N)
    q = gcd(ar2 + 1, N)
    if 1 < p < N:
        return p, N//p
    if 1 < q < N:
        return q, N//q
    return None

def qft_matrix(t):
    dim = 1 << t
    omega = np.exp(2j * np.pi / dim)
    j = np.arange(dim)
    k = j.reshape((dim,1))
    M = omega ** (k * j) / np.sqrt(dim)
    return M

def iqft_matrix(t):
    return np.conjugate(qft_matrix(t)).T

def simulate_shor_once(N, a, t=None, shots=1024, seed=None):
    """
    Simula la subrutina de búsqueda de periodo de Shor (didáctico).
    - N: entero compuesto a factorizar
    - a: base (debe ser coprima con N)
    - t: qubits para el registro de fase (si None usa 2*ceil(log2(N)))
    - shots: cantidad de muestras
    - seed: semilla para reproducibilidad
    Devuelve un dict con conteos, estimaciones de r y factores (si se encuentran).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    if gcd(a, N) != 1:
        return {"error": f"a={a} not coprime; gcd={gcd(a,N)}", "a": a}
    if t is None:
        t = 2 * ceil(log2(N))
    dim_phase = 1 << t
    m = ceil(log2(N))
    dim_target = 1 << m
    total_dim = dim_phase * dim_target

    # Construir vector de amplitudes tras la exponenciación modular
    state = np.zeros(total_dim, dtype=complex)
    for x in range(dim_phase):
        y = pow(a, x, N)
        idx = x * dim_target + y
        state[idx] = 1/np.sqrt(dim_phase)
    state_matrix = state.reshape((dim_phase, dim_target))

    # Aplicar IQFT sobre el registro de fase
    iqft = iqft_matrix(t)
    new_matrix = iqft @ state_matrix

    # Probabilidades sobre el registro de fase
    probs_phase = np.sum(np.abs(new_matrix)**2, axis=1)
    probs_phase = np.real_if_close(probs_phase)
    probs_phase = np.maximum(probs_phase, 0)
    probs_phase = probs_phase / np.sum(probs_phase)

    # Muestras (shots)
    samples = np.random.choice(np.arange(dim_phase), size=shots, p=probs_phase)
    unique, counts = np.unique(samples, return_counts=True)
    counts_dict = dict(zip(unique.tolist(), counts.tolist()))

    # Analizar los resultados más frecuentes para extraer candidatos r
    sorted_counts = sorted(counts_dict.items(), key=lambda kv: kv[1], reverse=True)
    K = min(8, len(sorted_counts))
    estimates = []
    for i in range(K):
        measured, cnt = sorted_counts[i]
        phase = measured / dim_phase
        frac = continued_fraction_expansion(phase, max_denominator=N)
        r_candidate = frac.denominator
        found_r = None
        for mult in range(1, 8):
            r_try = r_candidate * mult
            if pow(a, r_try, N) == 1:
                found_r = r_try
                break
        estimates.append({
            "measured": int(measured),
            "counts": int(cnt),
            "phase": float(phase),
            "fraction": f"{frac.numerator}/{frac.denominator}",
            "r_candidate": int(r_candidate),
            "r_found": int(found_r) if found_r is not None else None
        })

    # Intentar extraer factores a partir de las r encontradas
    factors = None
    for est in estimates:
        if est["r_found"] is None:
            continue
        r = est["r_found"]
        f = get_factors_from_r(a, r, N)
        if f is not None:
            factors = {"a": a, "r": r, "factors": f}
            break

    return {
        "N": N,
        "a": a,
        "t": t,
        "dim_phase": dim_phase,
        "dim_target": dim_target,
        "counts": counts_dict,
        "estimates": estimates,
        "factors": factors
    }
