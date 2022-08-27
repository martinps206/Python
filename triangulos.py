import math

# Modulo donde ingresamos las tres medidas de un triangulo, y nos 
# da como resultado un booleano, si es o no un triangulo Rectangulo.
def trianguloRectangulo(a,b,c):
    return c**2 == a**2 + b**2 

# Modulo para obtener el area de un triangulo por la formula de Heron
def area(a,b,c):
    s = (a+b+c)/2
    return math.sqrt(s*(s-a)*(s-b)*(s-c))

# Modulo que recibe las medidas de los tres lados del triangulo, 
# que por la definicion de triangulo devuelve el resultado en booleano
def esUnTriangulo(a, b, c):
    return a + b > c and b + c > a and c + a > b

# Modulo que sirve como menu, para poner en prueba nuestros modulos anteriores
def ingresarDatos():
    tri = []
    a = float(input("Ingresa la longitud del primer lado: "))
    
    b = float(input("Ingresa la longitud del segundo lado: "))
    if(a > b):
        x = b
        b = a
        a = x
    
    c = float(input("Ingresa la longitud del tercer lado: "))
    if(b > c):
        y = c
        c = b
        b = y
    tri.append(a)
    tri.append(b)
    tri.append(c)    
    if esUnTriangulo(a, b, c):
        print()
        print("Los lados del triangulo ingresado son : ")
        print(tri)
        print("Felicidades... Los datos ingresado cumplen con la definicion de una triangulo.")
        print()
        print("El area es : ",area(a,b,c))
        if(trianguloRectangulo(a,b,c)):
            print("Es un triangulo rectangulo...")
    else:
        print("Lo siento, no puede ser un triángulo.")


print("Analisis de  un triangulo")
print("-------------------------")
opt = int(input("Si deseas analizar tu triangulo ingresa 1, y si quieres salir delprograma ingresa 0\n"))
while (opt == 1):
    ingresarDatos()
    opt = int(input("Si deseas analizar tu triangulo ingresa 1, y si quieres salir delprograma ingresa 0\n"))
print("Muchas gracias...")