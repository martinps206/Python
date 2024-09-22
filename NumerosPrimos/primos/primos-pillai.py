import math

# Función para verificar si un número es primo
def es_primo(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Función para calcular n! (factorial)
def factorial(n):
    return math.prod(range(1, n+1))

# Función para verificar si un número primo p es de Pillai
def es_primo_pillai(p, max_n=10000):
    if not es_primo(p):
        return False
    for n in range(2, max_n):
        fact_n = factorial(n)
        if fact_n % (p ** 2) == (p ** 2) - 1:
            return True
    return False

# Búsqueda de números primos de Pillai en un rango dado
def buscar_primos_pillai(limite):
    primos_pillai = []
    for p in range(2, limite):
        if es_primo_pillai(p):
            primos_pillai.append(p)
    return primos_pillai

# Búsqueda de números primos de Pillai hasta el 1000
primos_pillai = buscar_primos_pillai(1000)
print(f"Números primos de Pillai hasta el 1000: {primos_pillai}")
