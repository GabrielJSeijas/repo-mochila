"""
Genera las instancias grandes de la segunda entrega con el esquema de Pisinger
("Core problems in knapsack algorithms", 1999; el mismo de las colecciones large_scale):

  tipo 1  no correlacionadas         p, w ~ U[1, R]
  tipo 2  debilmente correlacionadas w ~ U[1, R],  p ~ U[w - R/10, w + R/10]  (p >= 1)
  tipo 3  fuertemente correlacionadas w ~ U[1, R], p = w + R/10
  tipo 5  casi fuertemente corr.     w ~ U[1, R],  p ~ U[w + R/10 - R/500, w + R/10 + R/500]

  capacidad c = floor(suma(w) / 2)

Se escriben en inst/generadas/ con el formato de Ortega (primera linea "n c", luego "p w").
Uso: python3 generar.py
"""
import os

import numpy as np

TIPOS = (1, 2, 3, 5)
TAMANOS = (10_000, 25_000, 50_000, 100_000)
R = 100_000
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inst", "generadas")


def generar(tipo, n, R, semilla):
    rng = np.random.default_rng(semilla)
    w = rng.integers(1, R + 1, size=n)
    if tipo == 1:
        p = rng.integers(1, R + 1, size=n)
    elif tipo == 2:
        p = np.maximum(1, w + rng.integers(-R // 10, R // 10 + 1, size=n))
    elif tipo == 3:
        p = w + R // 10
    elif tipo == 5:
        p = w + R // 10 + rng.integers(-R // 500, R // 500 + 1, size=n)
    else:
        raise ValueError(tipo)
    c = int(w.sum()) // 2
    return p, w, c


def nombre(tipo, n, R):
    return "gen_%d_%d_%d" % (tipo, n, R)


def main():
    os.makedirs(DIR, exist_ok=True)
    for t in TIPOS:
        for n in TAMANOS:
            p, w, c = generar(t, n, R, semilla=1000 * t + n // 1000)
            with open(os.path.join(DIR, nombre(t, n, R)), "w") as fh:
                fh.write("%d %d\n" % (n, c))
                for a, b in zip(p, w):
                    fh.write("%d %d\n" % (a, b))
            print(nombre(t, n, R), "c =", c)


if __name__ == "__main__":
    main()
