def bellman_ford(grafo, inicio):
    dist = {v: float('inf') for v in grafo}
    dist[inicio] = 0

    for _ in range(len(grafo) - 1):
        for u in grafo:
            for v, peso in grafo[u].items():
                if dist[u] + peso < dist[v]:
                    dist[v] = dist[u] + peso

    # Verificación de ciclos negativos
    for u in grafo:
        for v, peso in grafo[u].items():
            if dist[u] + peso < dist[v]:
                raise ValueError("El grafo contiene un ciclo de peso negativo")

    return dist

# Ejemplo de uso
grafo = {
    'A': {'B': 1, 'C': 10, 'D': 4},
    'B': {'C': 3},
    'C': {'D': 2},
    'D': {'C': 2}
}

distancias = bellman_ford(grafo, 'A')
print(distancias)
