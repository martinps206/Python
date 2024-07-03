import heapq

class Grafo:
    def __init__(self):
        self.vertices = {}

    def agregar_vertice(self, nombre):
        self.vertices[nombre] = []

    def agregar_arista(self, desde, hacia, peso):
        self.vertices[desde].append((hacia, peso))
        self.vertices[hacia].append((desde, peso))  # Para grafos no dirigidos

    def dijkstra(self, inicio):
        distancias = {vertice: float('infinity') for vertice in self.vertices}
        distancias[inicio] = 0
        pq = [(0, inicio)]  # Priority queue

        while pq:
            distancia_actual, vertice_actual = heapq.heappop(pq)

            if distancia_actual > distancias[vertice_actual]:
                continue

            for vecino, peso in self.vertices[vertice_actual]:
                distancia = distancia_actual + peso

                if distancia < distancias[vecino]:
                    distancias[vecino] = distancia
                    heapq.heappush(pq, (distancia, vecino))

        return distancias

# Ejemplo de uso
g = Grafo()
g.agregar_vertice("A")
g.agregar_vertice("B")
g.agregar_vertice("C")
g.agregar_vertice("D")

g.agregar_arista("A", "B", 1)
g.agregar_arista("B", "C", 3)
g.agregar_arista("A", "C", 10)
g.agregar_arista("A", "D", 4)
g.agregar_arista("D", "C", 2)

distancias = g.dijkstra("A")
print(distancias)
