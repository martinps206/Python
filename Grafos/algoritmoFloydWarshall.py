def floyd_warshall(grafo):
    dist = {}
    for u in grafo:
        dist[u] = {}
        for v in grafo:
            if u == v:
                dist[u][v] = 0
            elif v in grafo[u]:
                dist[u][v] = grafo[u][v]
            else:
                dist[u][v] = float('inf')
    
    for k in grafo:
        for i in grafo:
            for j in grafo:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

# Ejemplo de uso
grafo = {
    'A': {'B': 1, 'C': 10, 'D': 4},
    'B': {'C': 3},
    'C': {'D': 2},
    'D': {'C': 2}
}

distancias = floyd_warshall(grafo)
print(distancias)
