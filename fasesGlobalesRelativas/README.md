# Fases globales y relativas — Demostración en Python

Este repositorio contiene una demostración educativa (en Python) sobre la diferencia entre fase global y fase relativa en qubits de un solo qubit. El ejemplo incluye cálculos simbólicos y numéricos, visualización en la esfera de Bloch y funciones reutilizables para experimentar.

## Contenido

- `fasesGlobalesRelativas.py` — Script principal que:
  - Muestra por qué una fase global no afecta las probabilidades de medición (simbolicamente y numéricamente).
  - Calcula probabilidades en la base computacional (Z) y en la base X (|+>, |->) para estados con y sin fase global.
  - Presenta un ejemplo de fase relativa que sí afecta resultados en la base X.
  - Grafica la esfera de Bloch y coloca los estados relevantes.
  - Exporta un archivo .npz con algunos vectores de estado de ejemplo.

## Objetivo pedagógico

Explicar y demostrar, con ejemplos y visualizaciones, que:

- Una fase global e^{iθ} aplicada a todo el vector de estado no cambia las probabilidades de medición en ninguna base. Matemáticamente, |e^{iθ} a|^2 = |a|^2.
- Una fase relativa entre amplitudes (por ejemplo, entre |0> y |1>) sí puede cambiar las probabilidades en bases diferentes a la base Z.

## Requisitos (dependencias)

- Python 3.8+ (recomendado 3.10+)
- numpy
- sympy
- matplotlib

Instalación rápida (recomendado crear un entorno virtual):

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy sympy matplotlib
```

Linux / macOS (bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy sympy matplotlib
```

Si prefieres usar pipx o conda, funciona igual.

## Uso

Ejecutar el script principal desde la raíz del repositorio:

PowerShell:

```powershell
python .\fasesGlobalesRelativas.py
```

El script imprimirá resultados numéricos y simbólicos en consola y abrirá una ventana con la esfera de Bloch mostrando los puntos relevantes. También intentará guardar un archivo `.npz` con algunos vectores de estado (la ruta usada por el script original es `/mnt/data/fases_global_relativa_demo.npz`; en Windows esa ruta puede no existir — ver nota de "Salida y guardado" más abajo).

## Funciones principales (API rápida)

El script define, además de la ejecución principal de demostración, las siguientes funciones reutilizables:

- `normalize(state: np.ndarray) -> np.ndarray` — Normaliza un vector de estado de 2 componentes.
- `apply_global_phase(state: np.ndarray, theta: float) -> np.ndarray` — Aplica una fase global e^{iθ} al estado.
- `prob_in_computational(state: np.ndarray)` — Devuelve (P(|0>), P(|1>)).
- `plus_minus_basis()` — Devuelve los vectores |+> y |->.
- `prob_in_basis(state: np.ndarray, basis_state: np.ndarray)` — Probabilidad de medir `basis_state` al medir `state`.
- `bloch_vector(state: np.ndarray)` — Calcula el vector de Bloch para un estado puro.
- `demonstrate_state(a, b, theta=0.0)` — Función empaquetada que devuelve un diccionario con probabilidades y vectores de Bloch para un par (a, b) y una fase global `theta`.

### Ejemplo interactivo en Python

Puedes importar la función desde el script (si lo conviertes a módulo) o copiarla en un REPL:

```python
from fasesGlobalesRelativas import demonstrate_state
res = demonstrate_state(np.sqrt(3)/2, 1/2, theta=1.23)
print(res['prob_Z'], res['prob_Z_global'])
```

## Notas sobre salida y guardado

- El script original guarda un `.npz` en `/mnt/data/...`. En Windows esa ruta probablemente no exista. Si quieres guardar localmente en Windows, modifica la línea final del script:

```python
np.savez('fases_global_relativa_demo.npz', ...)
```

o establece una ruta absoluta válida en tu sistema.