import math
# Nos informa la cantidad de dividisores del numero ingresado
def cantDivisores(num):
    i = 1
    j = 0
    while(num >= i):
        if (num % i == 0):
            j+=1
        i+=1
    return j

# Si el numero ingresado esta en el intervalo de (1,3)
def esPrimo(y): 
    return y > 1 and y < 3

# Nos informa si es primo o no es primo (0 : si es primo; 1 : no es primo) 
def es_primo (n):
    if(n==2): return 1
    
    if(n%2==0): return 0
    
    for i in range(1,n):
        if (n%i==0): return 0
    return 1

v = [1,2,3,4,5,6,7,8,9]
for i in range (9):
    print(v[i]," --> ",cantDivisores(v[i])," --> ",es_primo(cantDivisores(v[i])))

vector = []
for j in range (100):
    if(esPrimo(cantDivisores(j))):
        vector.append(j)
print(vector)        