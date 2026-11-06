"""
Experimentos de la segunda entrega.  Escribe resultados2.csv (una fila por instancia y
algoritmo, con los valores y tiempos de cada ejecucion; las tablas salen de tablas_e2.py).

Benchmark (36 instancias):
  B) large_scale de Ortega: 12 instancias, n = 100..10000, optimo publicado
  H) Pisinger (2005), tipos 11..16 con n = 10000, instancias 1 y 2: 12 instancias,
     optimo publicado
  G) generadas por nosotros (generar.py), tipos 1, 2, 3 y 5, n = 10 000, 25 000 y
     50 000, R = 100 000: 12 instancias.  Capacidades de 2.5e8 a 1.25e9: la PD no cabe en
     memoria.  Referencia: el optimo de OR-Tools si lo demuestra en 10 minutos; si no, el
     mejor valor conocido (el maximo entre OR-Tools y todas nuestras ejecuciones).

Protocolo (igual que en la primera entrega):
  * timeout de 10 minutos por ejecucion;
  * deterministas (PD, voraz, BL desde el voraz): una ejecucion, tiempo = minimo de 3
    repeticiones (solo si la ejecucion tarda menos de 5 s);
  * aleatorios (BL desde aleatorio, ILS, tabu, recocido, GRASP, genetico): 10 semillas,
    se reportan minimo, maximo y promedio del valor, del gap y del tiempo.

Uso:  python3 experimentos2.py [principal|parametros] [procesos] [filtro]
(los resultados parciales quedan en res/; si se interrumpe, se puede volver a correr)
"""
import csv
import glob
import os
import statistics as st
import sys
import time
from multiprocessing import Pool

import numpy as np

import etapa2 as e
import mochila as m

SEMILLAS = list(range(10))
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inst")


# ---------------------------------------------------------------------------
#  Instancias
# ---------------------------------------------------------------------------
def leer_generada(ruta):
    with open(ruta) as fh:
        filas = [l.split() for l in fh if l.strip()]
    n, c = int(filas[0][0]), int(filas[0][1])
    p = [int(a) for a, _ in filas[1:n + 1]]
    w = [int(b) for _, b in filas[1:n + 1]]
    return m.Instancia(os.path.basename(ruta), p, w, c, None)


def lista_instancias():
    """(grupo, nombre) de las 40 instancias, en el orden de las tablas."""
    lst = []
    for t in (1, 2):
        for n in (100, 200, 500, 1000, 2000, 5000, 10000):
            nom = "knapPI_%d_%d_1000_1" % (t, n)
            if os.path.exists(os.path.join(BASE, "large_scale", nom)):
                lst.append(("B", nom))
    for t in range(11, 17):
        for k in (1, 2):
            lst.append(("H", "knapPI_%d_10000_1000_%d" % (t, k)))
    for t in (1, 2, 3, 5):
        for n in (10_000, 25_000, 50_000, 100_000):
            lst.append(("G", "gen_%d_%d_100000" % (t, n)))
    return lst


def cargar(grupo, nom):
    if grupo == "B":
        return m.leer_ortega(os.path.join(BASE, "large_scale", nom),
                             os.path.join(BASE, "large_scale-optimum", nom), nom)
    if grupo == "H":
        t, n, k = nom.split("_")[1], nom.split("_")[2], int(nom.split("_")[4])
        return m.leer_pisinger(os.path.join(BASE, "pisinger_hard", "knapPI_%s_%s_1000.csv" % (t, n)),
                               cuantas=k)[k - 1]
    return leer_generada(os.path.join(BASE, "generadas", nom))


# ---------------------------------------------------------------------------
#  Algoritmos (nombre -> funcion(inst, semilla) que devuelve x)
# ---------------------------------------------------------------------------
def _x(r):
    return r[0] if isinstance(r, tuple) else r


ALEATORIOS = {
    "bl_aleat": lambda I, s, **k: e.bl_aleatoria(I, s, **k),
    "ils": lambda I, s, **k: m.ils(I, semilla=s, **k),
    "tabu": lambda I, s, **k: e.tabu(I, semilla=s, **k),
    "recocido": lambda I, s, **k: e.recocido(I, semilla=s, **k),
    "grasp": lambda I, s, **k: e.grasp(I, semilla=s, **k),
    "genetico": lambda I, s, **k: e.genetico(I, semilla=s, **k),
}

# variantes de parametros (la primera de cada tecnica es la configuracion por defecto)
VARIANTES = {
    "tabu": [("t=10", {}), ("t=5", {"tenencia": 5}), ("t=30", {"tenencia": 30})],
    "recocido": [("a=0.95", {}), ("a=0.90", {"alfa": 0.90}), ("a=0.99", {"alfa": 0.99})],
    "grasp": [("a=0.97", {}), ("a=0.9", {"alfa": 0.9}), ("a=0.99", {"alfa": 0.99}),
              ("reactivo", {"reactivo": True})],
    "genetico": [("unif,N=50", {}), ("1pto,N=50", {"cruce": "un_punto"}),
                 ("unif,N=20", {"tam_pob": 20}), ("unif,pm=10/n", {"prob_mut": "10/n"})],
}


def cronometrar(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t


# ---------------------------------------------------------------------------
#  Una instancia completa (se corre en un proceso aparte)
# ---------------------------------------------------------------------------
MEM_ORTOOLS = 3 * 2**30    # bytes de memoria virtual que le damos a OR-Tools


_SCRIPT_ORTOOLS = """
import sys, time
from ortools.algorithms.python import knapsack_solver as ks
lim = float(sys.argv[1])
c = int(sys.stdin.readline())
p = [int(v) for v in sys.stdin.readline().split()]
w = [int(v) for v in sys.stdin.readline().split()]
s = ks.KnapsackSolver(ks.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, "ref")
s.init(p, [w], [c])
s.set_time_limit(lim)
v = s.solve()
print(int(v), int(s.is_solution_optimal()))
"""


def referencia_ortools(I, limite=600):
    """OR-Tools en un proceso aparte con la memoria limitada: su arbol de ramificacion
    crece sin control en las instancias dificiles y, sin limite, se come toda la RAM.
    Devuelve (valor o None, estado, tiempo) con estado optimo / no_demostrado / memoria."""
    import resource
    import subprocess

    def limitar():
        resource.setrlimit(resource.RLIMIT_AS, (MEM_ORTOOLS, MEM_ORTOOLS))

    entrada = "%d\n%s\n%s\n" % (int(I.c), " ".join(map(str, I.p.tolist())),
                                  " ".join(map(str, I.w.tolist())))
    t = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, "-c", _SCRIPT_ORTOOLS, str(limite)], input=entrada,
                           capture_output=True, text=True, preexec_fn=limitar, timeout=limite + 120)
        t = time.perf_counter() - t
        if r.returncode == 0 and r.stdout.strip():
            v, opt = r.stdout.split()
            return int(v), ("optimo" if opt == "1" else "no_demostrado"), t
    except subprocess.TimeoutExpired:
        t = time.perf_counter() - t
    return None, "memoria", t


def correr_instancia(arg):
    grupo, nom = arg
    I = cargar(grupo, nom)
    filas = []

    def fila(alg, variante, valores, tiempos, extra=""):
        filas.append(dict(grupo=grupo, instancia=nom, n=I.n, c=I.c, algoritmo=alg,
                          variante=variante, valores=" ".join("%d" % v for v in valores),
                          tiempos=" ".join("%.6f" % t for t in tiempos), extra=extra))

    # --- PD
    (est, xe, ve), te = cronometrar(m.exacto_detalle, I)
    if est == "ok":
        assert m.es_factible(I, xe)
    fila("pd", est, [] if ve is None else [ve], [te])
    # --- referencia externa para G
    ref_extra = ""
    if grupo == "G":
        v, est_o, t = referencia_ortools(I)
        fila("ortools", est_o, [] if v is None else [v], [t])
    # --- deterministas
    for alg, fn in (("voraz", lambda: m.voraz(I)),
                    ("bl_voraz", lambda: m.busqueda_local(I, m.voraz(I))[0])):
        x, t = cronometrar(fn)
        if t < 5:
            t = min([t] + [cronometrar(fn)[1] for _ in range(2)])
        assert m.es_factible(I, x)
        fila(alg, "", [m.valor(I, x)], [t])
    fila("cota_dantzig", "", [m.cota_dantzig(I)], [0])
    # --- aleatorios, configuracion por defecto
    for alg, fn in ALEATORIOS.items():
        vals, ts = [], []
        for s in SEMILLAS:
            r, t = cronometrar(fn, I, s)
            x = _x(r)
            assert m.es_factible(I, x), (nom, alg, s)
            vals.append(m.valor(I, x))
            ts.append(t)
        fila(alg, "defecto", vals, ts)
    guardar(filas, "principal", nom)
    print("listo", nom, flush=True)
    return filas


def correr_parametros(arg):
    grupo, nom = arg
    I = cargar(grupo, nom)
    filas = []
    for alg, vs in VARIANTES.items():
        for etiqueta, kw in vs[1:]:
            kw = dict(kw)
            if kw.get("prob_mut") == "10/n":
                kw["prob_mut"] = 10.0 / I.n
            vals, ts = [], []
            for s in SEMILLAS:
                r, t = cronometrar(ALEATORIOS[alg], I, s, **kw)
                x = _x(r)
                assert m.es_factible(I, x)
                vals.append(m.valor(I, x))
                ts.append(t)
            filas.append(dict(grupo=grupo, instancia=nom, n=I.n, c=I.c, algoritmo=alg,
                              variante=etiqueta, valores=" ".join("%d" % v for v in vals),
                              tiempos=" ".join("%.6f" % t for t in ts), extra=""))
    guardar(filas, "parametros", nom)
    print("listo (parametros)", nom, flush=True)
    return filas


def guardar(filas, modo, nom):
    os.makedirs(os.path.join("res", modo), exist_ok=True)
    with open(os.path.join("res", modo, nom + ".csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "principal"
    trabajos = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    insts = lista_instancias()
    if len(sys.argv) > 3:                       # filtro opcional por nombre
        insts = [g for g in insts if any(s in g[1] for s in sys.argv[3].split(","))]
    # las mas pesadas primero, para repartir mejor el trabajo
    insts.sort(key=lambda g: -cargar(*g).n)
    fn = correr_instancia if modo == "principal" else correr_parametros
    # se puede retomar: las instancias que ya tienen archivo en res/<modo>/ no se repiten
    pend = [g for g in insts if not os.path.exists(os.path.join("res", modo, g[1] + ".csv"))]
    with Pool(trabajos) as pool:
        for _ in pool.imap_unordered(fn, pend):
            pass
    # se unen los resultados en un solo CSV
    salida = "resultados2.csv" if modo == "principal" else "parametros2.csv"
    filas = []
    for g in lista_instancias():
        ruta = os.path.join("res", modo, g[1] + ".csv")
        if os.path.exists(ruta):
            filas += list(csv.DictReader(open(ruta)))
    with open(salida, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)


if __name__ == "__main__":
    main()
