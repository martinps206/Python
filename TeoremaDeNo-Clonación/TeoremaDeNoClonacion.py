from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

print("Aer está disponible:", Aer)

# Crear un circuito cuántico con 3 qubits (1 para el estado original, 1 para el intento de clonación, y 1 auxiliar)
qc = QuantumCircuit(3, 2)

# Preparar un estado arbitrario en el primer qubit (estado original)
qc.h(0)  # Aplicamos una puerta Hadamard para crear una superposición (|+>)

# Intentar clonar el estado aplicando puertas CNOT (esto en realidad no clonará correctamente el estado)
qc.cx(0, 1)  # Intento de copiar el estado del primer qubit al segundo

# Medir los resultados para ver si el estado fue clonado
qc.measure([0, 1], [0, 1])

# Simulación del circuito cuántico
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qc, simulator)
qobj = assemble(compiled_circuit)
result = simulator.run(qobj).result()

# Obtener los resultados
counts = result.get_counts(qc)
print("Resultados de la simulación:", counts)

# Visualizar el histograma de los resultados
plot_histogram(counts)
plt.show()
