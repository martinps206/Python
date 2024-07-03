import heapq

def a_star(grafo, inicio, objetivo, heuristica):
    pq = [(0, inicio)]
    dist = {inicio: 0}
    padres = {inicio: None}

    while pq:
        _, actual = heapq.heappop(pq)

        if actual == objetivo:
            path = []
            while actual:
                path.append(actual)
                actual = padres[actual]
            return path[::-1]

        for vecino, peso in grafo[actual].items():
            nueva_dist = dist[actual] + peso

            if vecino not in dist or nueva_dist < dist[vecino]:
                dist[vecino] = nueva_dist
                prioridad = nueva_dist + heuristica(vecino, objetivo)
                heapq.heappush(pq, (prioridad, vecino))
                padres[vecino] = actual

    return None

# Ejemplo de uso
grafo = {
    'A': {'B': 1, 'C': 10, 'D': 4},
    'B': {'C': 3},
    'C': {'D': 2},
    'D': {'C': 2}
}

def heuristica(u, v):
    # Supongamos una heurística trivial
    return 0

camino = a_star(grafo, 'A', 'C', heuristica)
print(camino)
