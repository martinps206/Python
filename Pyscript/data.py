import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(10, 6))

# pais 1
x = [2016, 2017, 2018, 2019, 2020, 2021] #años
y = [42, 43, 45, 47, 48, 50] #poblacion

# pais 2
x2 = [2016, 2017, 2018, 2019, 2020, 2021] #años
y2 = [43, 43, 44, 44, 45, 45] #poblacion

# plotting
plt.plot(x, y, marker='o', linestyle='--', color='g', label='Peru')
plt.plot(x2, y2, marker='d', linestyle='-', color='r', label='Argentina')

# agregar nombre a los ejes y titulo al grafico
plt.xlabel('Años')
plt.ylabel('Poblacion (M)')
plt.title('Años vs Poblacion')
plt.legend(loc='lower right')

plt.yticks([41, 45, 48, 51])

# guardar figura
plt.savefig('ejemplo1.png')

# mostrar plot
plt.show()


# ejemplo 2 subplots
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 4)) #figsize modifica el tamaño del plot 

ax[0].plot(x, y, color='g', label='Peru')
ax[0].legend()

ax[1].plot(x2, y2, color='r', label='Argentina')
ax[1].legend()
plt.savefig('ejemplo2.png')
plt.show()

# ------
# ejemplo +2 subplots

# fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

# ax[0][0].plot(x, y, 'o-', c='r', label='Colombia') # fila 1 - columna 0 (columnas comienzan en 0)
# ax[0][2].plot(x2, y2, 'v--', c='b', label='Argentina') # fila 1 - columna 2

# plt.show()


x1 = ['Argentina', 'Colombia', 'Peru'] # paises 
y1 = [40, 50, 47] # poblacion (data numérica)

plt.bar(x1, y1)
plt.savefig('ejemplo3.png')
plt.show()

a = [1, 2, 3, 4, 5, 4, 3 ,2 ,2, 4, 5, 6, 7]
b = [7, 2, 3, 5, 5, 7, 3, 2, 1, 4, 6, 3, 2]

plt.scatter(a, b)
plt.savefig('ejemplo4.png')
plt.show()