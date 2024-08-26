import numpy as np
from scipy.optimize import fsolve

# Función de oferta: q_o(p) = (3p + 1800) / 200
def oferta(p):
    return (3 * p + 1800) / 200

# Función de demanda: q_d(p) = 600 - (100p / 3)
def demanda(p):
    return 600 - (100 * p / 3)

# Generar valores de p para el tramo racionalmente económico
p_values = np.linspace(0, 18, 100)

# Calcular valores correspondientes de q para oferta y demanda
q_oferta = oferta(p_values)
q_demanda = demanda(p_values)

# Mostrar el tramo racionalmente económico (0 < p < 18)
print("Tramo racionalmente económico:")
print(f"Oferta: q_o(p) = (3p + 1800) / 200")
print(f"Demanda: q_d(p) = 600 - (100p / 3)")

# Función de oferta ajustada con el impuesto: q_o'(p) = (3p + 1799.19) / 200
def oferta_ajustada(p):
    return (3 * p + 1799.19) / 200

# Encontrar el punto de equilibrio con el impuesto
def equilibrio(p):
    return oferta_ajustada(p) - demanda(p)

# Usar fsolve para encontrar el valor de p que satisface la ecuación de equilibrio
p_equilibrio = fsolve(equilibrio, 10)[0]  # Iniciamos la búsqueda en p = 10
q_equilibrio = oferta_ajustada(p_equilibrio)

print(f"Punto de equilibrio con impuesto:")
print(f"Precio de equilibrio (p'): {p_equilibrio:.2f} soles")
print(f"Cantidad de equilibrio (q'): {q_equilibrio:.2f} unidades")


# Ingreso antes del impuesto
p_original = 18  # Punto de equilibrio original
q_original = oferta(p_original)
ingreso_original = p_original * q_original

# Ingreso después del impuesto
ingreso_nuevo = (p_equilibrio - 0.27) * q_equilibrio

# Variación en el ingreso
variacion_ingreso = ingreso_nuevo - ingreso_original

print(f"Ingreso original del vendedor: {ingreso_original:.2f} soles")
print(f"Ingreso nuevo del vendedor: {ingreso_nuevo:.2f} soles")
print(f"Variación en el ingreso: {variacion_ingreso:.2f} soles")
