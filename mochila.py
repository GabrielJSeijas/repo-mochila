"""
Problema de la mochila 0/1 -- CI5652 Diseno de Algoritmos II, Proyecto (Primer corte).

Cambios para el segundo corte (la logica de los algoritmos es la misma):
  * exacto_detalle: la PD revisa antes cuanta memoria necesita y dice por que no hay solucion
    ('timeout' o 'memoria'); si no cabe la tabla de reconstruccion, calcula solo el valor.
  * busqueda_local / mejor_vecino: los objetos se ordenan por peso una sola vez por busqueda
    y las busquedas binarias se hacen con consultas ordenadas (mismo resultado, mas rapido).

Contiene:
  * lectura de instancias (formato de Ortega "instances_01_KP" y formato de Pisinger 2005)
  * solucion exacta por programacion dinamica (pseudo-polinomial, O(n*c))
  * heuristica voraz por razon valor/peso (con la correccion de 1/2-aproximacion)
  * busqueda local (vecindad AGREGAR + INTERCAMBIO 1-1, mejor vecino)
  * busqueda local iterada (perturbacion: sacar k objetos + rellenar sin ellos)

Timeout: todas las funciones de busqueda reciben `limite` (segundos, por defecto
TIMEOUT = 600 s = 10 min).  La PD devuelve None si no termina a tiempo ("sin solucion");
la busqueda local y la ILS devuelven la mejor solucion encontrada hasta ese momento.

Una solucion se representa como un vector binario x (numpy bool), x[i] = True si el
objeto i esta en la mochila. Todas las soluciones manejadas son factibles.
"""
import math
import random
import time
import numpy as np

TIMEOUT = 600.0          # 10 minutos por ejecucion de cada algoritmo


def _fin(limite):
    """Instante (perf_counter) en que se acaba el tiempo."""
    return time.perf_counter() + (TIMEOUT if limite is None else limite)


# ---------------------------------------------------------------------------
#  Instancias
# ---------------------------------------------------------------------------
class Instancia:
    def __init__(self, nombre, p, w, c, optimo=None):
        self.nombre = nombre
        p, w = np.asarray(p, dtype=float), np.asarray(w, dtype=float)
        self.entera = bool(np.all(p == np.round(p)) and np.all(w == np.round(w)) and float(c).is_integer())
        tipo = np.int64 if self.entera else float
        self.p, self.w = p.astype(tipo), w.astype(tipo)
        self.c = int(c) if self.entera else float(c)
        self.n = len(self.p)
        self.optimo = optimo            # valor optimo conocido (si viene con el benchmark)


def leer_ortega(ruta, ruta_optimo, nombre):
    """Formato 'instances_01_KP': primera linea 'n c', luego 'p w' por objeto."""
    with open(ruta) as fh:
        filas = [l.split() for l in fh if l.strip()]
    n, c = int(filas[0][0]), float(filas[0][1])
    p = [float(a) for a, _ in filas[1:n + 1]]
    w = [float(b) for _, b in filas[1:n + 1]]
    with open(ruta_optimo) as fh:
        opt = float(fh.read().split()[0])
    return Instancia(nombre, p, w, c, opt)


def leer_pisinger(ruta, cuantas=None):
    """Formato de Pisinger (2005): varias instancias por archivo separadas por '-----'."""
    instancias, bloque = [], []
    with open(ruta) as fh:
        for linea in fh:
            linea = linea.strip()
            if linea.startswith("-----"):
                if bloque:
                    instancias.append(_bloque_pisinger(bloque))
                    bloque = []
                    if cuantas and len(instancias) >= cuantas:
                        break
            elif linea:
                bloque.append(linea)
    if bloque and (not cuantas or len(instancias) < cuantas):
        instancias.append(_bloque_pisinger(bloque))
    return instancias


def _bloque_pisinger(b):
    nombre = b[0]
    n, c, z = int(b[1].split()[1]), int(b[2].split()[1]), int(b[3].split()[1])
    p, w = [], []
    for fila in b[5:5 + n]:
        _, pi, wi, _ = fila.split(",")
        p.append(int(pi))
        w.append(int(wi))
    return Instancia(nombre, p, w, c, z)


# ---------------------------------------------------------------------------
#  Utilidades
# ---------------------------------------------------------------------------
def valor(inst, x):
    return inst.p[x].sum().item()


def peso(inst, x):
    return inst.w[x].sum().item()


def razon(inst):
    """p_i / w_i (un objeto de peso 0 tiene razon infinita: siempre conviene)."""
    with np.errstate(divide="ignore"):
        return np.where(inst.w > 0, inst.p / np.where(inst.w > 0, inst.w, 1), np.inf)


def es_factible(inst, x):
    return peso(inst, x) <= inst.c


def cota_dantzig(inst):
    """Cota superior de la relajacion lineal (Dantzig, 1957): voraz + fraccion del objeto critico."""
    orden = np.argsort(-razon(inst), kind="stable")
    cap, tot = inst.c, 0.0
    for i in orden:
        if inst.w[i] <= cap:
            cap -= inst.w[i]
            tot += inst.p[i]
        else:
            tot += inst.p[i] * cap / inst.w[i]
            break
    return math.floor(tot) if inst.entera else tot


# ---------------------------------------------------------------------------
#  1) Solucion exacta: programacion dinamica O(n*c)
# ---------------------------------------------------------------------------
def exacto(inst, limite=None):
    """
    PD si los datos son enteros; si son reales (instancia f5) se enumeran los 2^n subconjuntos.
    Devuelve None si se acaba el tiempo (se reporta "sin solucion").
    """
    return exacto_pd(inst, limite) if inst.entera else exacto_enumeracion(inst, limite)


def exacto_enumeracion(inst, limite=None):
    """Fuerza bruta por bits para n pequeno (n <= 25): O(2^n * n)."""
    n = inst.n
    if n > 25:
        return None                       # 2^n subconjuntos no caben en el tiempo/memoria
    masc = np.arange(1 << n, dtype=np.int64)
    bits = ((masc[:, None] >> np.arange(n)) & 1).astype(bool)
    pes = bits @ inst.w
    val = np.where(pes <= inst.c + 1e-9, bits @ inst.p, -1)
    return bits[int(np.argmax(val))].copy()


def exacto_pd(inst, limite=None):
    """
    F[k][r] = mejor valor usando los primeros k objetos con capacidad r.
        F[k][r] = max(F[k-1][r], F[k-1][r - w_k] + p_k)   si w_k <= r
    Se guarda solo una fila de F (O(c) memoria) y un bit por (k, r) que indica si
    el objeto k se tomo, para poder reconstruir la solucion.
    """
    fin = _fin(limite)
    n, c = inst.n, inst.c
    F = np.zeros(c + 1, dtype=np.int64)
    tomo = np.zeros((n, (c + 1 + 7) // 8), dtype=np.uint8)     # bits empaquetados
    for k in range(n):
        if time.perf_counter() > fin:      # timeout: no se pudo encontrar la solucion
            return None
        wk, pk = int(inst.w[k]), int(inst.p[k])
        if wk > c:
            continue
        cand = F[:c + 1 - wk] + pk
        mejora = cand > F[wk:]
        bits = np.zeros(c + 1, dtype=bool)
        bits[wk:] = mejora
        F[wk:] = np.where(mejora, cand, F[wk:])
        tomo[k] = np.packbits(bits)
    # reconstruccion
    x = np.zeros(n, dtype=bool)
    r = c
    for k in range(n - 1, -1, -1):
        if (tomo[k, r >> 3] >> (7 - (r & 7))) & 1:
            x[k] = True
            r -= int(inst.w[k])
    return x


MEMORIA_MAX = 2.0e9      # bytes que le permitimos usar a la PD (la maquina tiene 8 GB)


def exacto_detalle(inst, limite=None):
    """
    Version de la PD usada en la segunda entrega, que dice POR QUE no hay solucion:
      'ok'          -> x optimo (PD con reconstruccion, o enumeracion si los datos son reales)
      'solo_valor'  -> la tabla de reconstruccion (n*c bits) no cabe; se calcula solo F(n,c)
      'timeout'     -> no termino en el tiempo limite (10 min)
      'memoria'     -> ni siquiera una fila de F (c+1 enteros) cabe en MEMORIA_MAX
    Devuelve (estado, x, valor).
    """
    if not inst.entera:
        x = exacto_enumeracion(inst, limite)
        return ("ok", x, valor(inst, x)) if x is not None else ("memoria", None, None)
    n, c = inst.n, inst.c
    fila = 8.0 * (c + 1) * 3                 # F mas dos temporales del tamano de una fila
    bits = n * (c + 1) / 8.0
    if fila > MEMORIA_MAX:
        return "memoria", None, None
    if fila + bits <= MEMORIA_MAX:
        x = exacto_pd(inst, limite)
        return ("ok", x, valor(inst, x)) if x is not None else ("timeout", None, None)
    # solo el valor: una fila, sin reconstruccion
    fin = _fin(limite)
    F = np.zeros(c + 1, dtype=np.int64)
    for k in range(n):
        if time.perf_counter() > fin:
            return "timeout", None, None
        wk, pk = int(inst.w[k]), int(inst.p[k])
        if wk > c:
            continue
        cand = F[:c + 1 - wk] + pk
        np.maximum(F[wk:], cand, out=F[wk:])
    return "solo_valor", None, int(F[c])


# ---------------------------------------------------------------------------
#  2) Heuristica especializada: voraz por razon valor/peso
# ---------------------------------------------------------------------------
def voraz(inst):
    """
    Ordena por p_i/w_i decreciente y agrega cada objeto que todavia quepa.
    Al final compara con la mejor solucion de un solo objeto: max(voraz, mejor objeto)
    garantiza al menos la mitad del optimo (1/2-aproximacion).
    """
    orden = np.argsort(-razon(inst), kind="stable")
    x = np.zeros(inst.n, dtype=bool)
    cap = inst.c
    for i in orden:
        if inst.w[i] <= cap:
            x[i] = True
            cap -= inst.w[i]
    cabe = inst.w <= inst.c
    if cabe.any():
        j = int(np.argmax(np.where(cabe, inst.p, -1)))
        if inst.p[j] > valor(inst, x):
            x = np.zeros(inst.n, dtype=bool)
            x[j] = True
    return x


# ---------------------------------------------------------------------------
#  3) Busqueda local: vecindad AGREGAR + INTERCAMBIO(1,1), mejor vecino
# ---------------------------------------------------------------------------
def mejor_vecino(inst, x, ordw=None):
    """
    Devuelve (delta, i, j) del mejor vecino de x, o None si ninguno mejora.
      AGREGAR(j):       x_j = 0 -> 1, si w_j <= r                  delta = p_j
      INTERCAMBIO(i,j): x_i = 1 -> 0 y x_j = 0 -> 1, si w_j <= r + w_i
                                                                     delta = p_j - p_i
    donde r = c - peso(x) es la capacidad residual (i = -1 indica AGREGAR).

    En lugar de revisar los |dentro| x |fuera| pares, se ordenan los objetos de fuera por
    peso y se precalcula el maximo prefijo de su valor: para cada i, el mejor j que cabe es
    el de mayor p_j con w_j <= r + w_i (busqueda binaria).  Costo O(n log n) por paso,
    y el resultado es exactamente el mejor vecino.  `ordw` (los objetos ordenados por peso)
    se calcula una sola vez por busqueda local.
    """
    r = inst.c - peso(inst, x)
    if ordw is None:                     # orden de todos los objetos por peso (se puede
        ordw = np.argsort(inst.w, kind="stable")   # calcular una sola vez por busqueda)
    orden = ordw[~x[ordw]]               # objetos de fuera, ordenados por peso
    if len(orden) == 0:
        return None
    w_ord = inst.w[orden]
    p_ord = inst.p[orden]
    idx_max = np.zeros(len(orden), dtype=np.int64)          # posicion del maximo prefijo
    m = np.maximum.accumulate(p_ord)
    nuevo = np.r_[True, m[1:] > m[:-1]]
    idx_max = np.maximum.accumulate(np.where(nuevo, np.arange(len(orden)), 0))

    dentro = ordw[x[ordw]]               # objetos de dentro, tambien ordenados por peso
    quitar = np.r_[-1, dentro]                               # -1 = no se quita nada (AGREGAR)
    w_q = np.r_[0, inst.w[dentro]]
    p_q = np.r_[0, inst.p[dentro]]
    # consultas ordenadas: la busqueda binaria es ~5 veces mas rapida en memoria
    tope = np.searchsorted(w_ord, r + w_q + 1e-9 * (not inst.entera), side="right") - 1
    valido = tope >= 0
    if not valido.any():
        return None
    ganancia = np.where(valido, m[np.maximum(tope, 0)] - p_q, -np.inf)
    gmax = ganancia.max()
    if gmax <= 0:
        return None
    # en caso de empate se escoge el de menor indice (AGREGAR primero), como en la entrega 1
    empates = np.flatnonzero(ganancia == gmax)
    k = int(empates[np.argmin(quitar[empates])])
    j = int(orden[idx_max[tope[k]]])
    return (ganancia[k].item(), int(quitar[k]), j)


def busqueda_local(inst, x, max_pasos=10**7, limite=None, fin=None):
    """Mejor vecino hasta que ninguno mejore; si se acaba el tiempo devuelve la solucion actual."""
    fin = _fin(limite) if fin is None else fin
    x = x.copy()
    pasos = 0
    ordw = np.argsort(inst.w, kind="stable")
    while pasos < max_pasos and time.perf_counter() <= fin:
        mov = mejor_vecino(inst, x, ordw)
        if mov is None:
            break
        _, i, j = mov
        if i >= 0:
            x[i] = False
        x[j] = True
        pasos += 1
    return x, pasos


# ---------------------------------------------------------------------------
#  4) Busqueda local iterada
# ---------------------------------------------------------------------------
def perturbar(inst, x, k, rng):
    """
    Saca k objetos de la mochila escogidos al azar y rellena con el voraz por razon,
    pero sin permitir volver a meter los k objetos sacados (memoria de corto plazo).
    Fuerza de la perturbacion: entre k y 2k posiciones de x cambian.
    """
    y = x.copy()
    dentro = np.flatnonzero(y)
    if len(dentro) == 0:
        return y
    k = min(k, len(dentro))
    sacados = rng.sample(list(dentro), k)
    y[sacados] = False
    prohibido = np.zeros(inst.n, dtype=bool)
    prohibido[sacados] = True
    cap = inst.c - peso(inst, y)
    orden = np.argsort(-razon(inst), kind="stable")
    for i in orden:
        if not y[i] and not prohibido[i] and inst.w[i] <= cap:
            y[i] = True
            cap -= inst.w[i]
    return y


def ils(inst, semilla=0, max_iter=200, max_sin_mejora=100, fuerza=0.05, limite=None):
    """
    ILS (Clase 3):
      S0  <- voraz;  S* <- BL(S0)
      repetir
         S'  <- perturbar(S*)        (k aleatorio en [1, max(2, fuerza*|S*|)])
         S'* <- BL(S')
         S*  <- S'* si f(S'*) >= f(S*)   (se aceptan empates para moverse en mesetas)
      hasta max_iter iteraciones, max_sin_mejora sin mejorar la mejor, alcanzar la cota de Dantzig
      o agotar el tiempo limite (se devuelve la mejor solucion encontrada hasta ese momento)
    """
    fin = _fin(limite)
    rng = random.Random(semilla)
    cota = cota_dantzig(inst)
    s, _ = busqueda_local(inst, voraz(inst), fin=fin)
    mejor, fmejor = s.copy(), valor(inst, s)
    fs = fmejor
    sin_mejora, it = 0, 0
    while (it < max_iter and sin_mejora < max_sin_mejora and fmejor < cota
           and time.perf_counter() <= fin):
        it += 1
        kmax = max(2, math.ceil(fuerza * int(s.sum())))
        k = rng.randint(1, kmax)
        s2, _ = busqueda_local(inst, perturbar(inst, s, k, rng), fin=fin)
        f2 = valor(inst, s2)
        if f2 >= fs:
            s, fs = s2, f2
        if f2 > fmejor:
            mejor, fmejor, sin_mejora = s2.copy(), f2, 0
        else:
            sin_mejora += 1
    return mejor, it
