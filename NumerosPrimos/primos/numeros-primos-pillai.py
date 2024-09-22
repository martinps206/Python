from sympy import factorial, isprime

# Función optimizada para verificar si p es un número primo de Pillai
def es_primo_pillai_optimizado(p, max_n=1000):
    if not isprime(p):
        return False
    
    for n in range(2, max_n):
        fact_n = factorial(n)
        if (fact_n % (p ** 2)) == -1 % (p ** 2):
            return True
    return False

# Función para buscar números primos de Pillai de manera optimizada
def buscar_primos_pillai_optimizado(limite):
    primos_pillai = []
    for p in range(2, limite):
        if isprime(p) and es_primo_pillai_optimizado(p):
            primos_pillai.append(p)
    return primos_pillai

# Búsqueda de primos de Pillai hasta el 1000 con el método optimizado
primos_pillai_opt = buscar_primos_pillai_optimizado(1000)
print(f"Números primos de Pillai optimizados hasta el 1000: {primos_pillai_opt}")
