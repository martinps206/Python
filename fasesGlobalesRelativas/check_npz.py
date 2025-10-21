import numpy as np
import os
import sys

f = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fases_global_relativa_demo.npz'))
if not os.path.exists(f):
    print('No encontrado:', f)
    sys.exit(2)

try:
    d = np.load(f, allow_pickle=True)
    print('Ruta:', f)
    print('Claves:', list(d.keys()))
    for k in d.files:
        v = d[k]
        try:
            shape = getattr(v, 'shape', None)
        except Exception:
            shape = None
        print(k, 'shape=', shape, 'dtype=', getattr(v, 'dtype', None))
except Exception as e:
    print('Error leyendo .npz:', e)
    sys.exit(1)
