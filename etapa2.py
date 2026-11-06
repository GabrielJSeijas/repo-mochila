"""
Problema de la mochila 0/1 -- CI5652 Diseno de Algoritmos II, Proyecto (Segundo corte).

Algoritmos nuevos de esta entrega (usan las utilidades de mochila.py):
  * busqueda local desde soluciones aleatorias (experimento pedido en la primera entrega)
  * busqueda tabu        (vecindad AGREGAR / QUITAR / INTERCAMBIO, atributo tabu = objeto movido)
  * recocido simulado    (vecino aleatorio, criterio de Metropolis, enfriamiento geometrico)
  * GRASP                (RCL por valor con parametro alfa sobre la razon p/w + busqueda local;
                          version normal y version reactiva)
  * algoritmo genetico   (genotipo = cadena binaria, reparacion voraz, torneo, cruce
                          uniforme o de un punto, mutacion bit a bit, elitismo)

Todas reciben `limite` (segundos; por defecto mochila.TIMEOUT = 10 min) y, si se acaba el
tiempo, devuelven la mejor solucion encontrada hasta ese momento.  Todas devuelven
(x, info), donde x es un vector booleano factible e info un diccionario con contadores.
"""
import bisect
import math
import random
import time

import numpy as np

import mochila as m


def _ahora():
    return time.perf_counter()


# ---------------------------------------------------------------------------
#  Busqueda local desde una solucion aleatoria
# ---------------------------------------------------------------------------
def solucion_aleatoria(inst, rng):
    """Recorre los objetos en un orden aleatorio y mete cada uno que quepa (solucion maximal)."""
    orden = list(range(inst.n))
    rng.shuffle(orden)
    x = np.zeros(inst.n, dtype=bool)
    cap = inst.c
    w = inst.w.tolist()
    for i in orden:
        if w[i] <= cap:
            x[i] = True
            cap -= w[i]
    return x


def bl_aleatoria(inst, semilla=0, limite=None):
    rng = random.Random(semilla)
    fin = m._fin(limite)
    x0 = solucion_aleatoria(inst, rng)
    x, pasos = m.busqueda_local(inst, x0, fin=fin)
    return x, {"inicial": m.valor(inst, x0), "pasos": pasos}


# ---------------------------------------------------------------------------
#  Vecindad para tabu: mejor movimiento entre los permitidos, aunque empeore
# ---------------------------------------------------------------------------
def mejor_movimiento(inst, x, permitido, ordw=None):
    """
    Igual que mochila.mejor_vecino, pero:
      * solo considera objetos con permitido[i] = True (los demas son tabu),
      * agrega el movimiento QUITAR(i)  (delta = -p_i),
      * devuelve el mejor movimiento aunque su delta sea negativo.
    Devuelve (delta, i, j) con i = -1 para AGREGAR(j) y j = -1 para QUITAR(i), o None.
    """
    r = inst.c - m.peso(inst, x)
    mejor = None
    if ordw is None:
        ordw = np.argsort(inst.w, kind="stable")
    orden = ordw[~x[ordw] & permitido[ordw]]      # fuera y permitidos, por peso
    dentro = ordw[x[ordw] & permitido[ordw]]      # dentro y permitidos, por peso
    if len(orden):
        w_ord, p_ord = inst.w[orden], inst.p[orden]
        mx = np.maximum.accumulate(p_ord)
        nuevo = np.r_[True, mx[1:] > mx[:-1]]
        idx_max = np.maximum.accumulate(np.where(nuevo, np.arange(len(orden)), 0))
        quitar = np.r_[-1, dentro]
        w_q = np.r_[0, inst.w[dentro]]
        p_q = np.r_[0, inst.p[dentro]]
        tope = np.searchsorted(w_ord, r + w_q, side="right") - 1
        valido = tope >= 0
        if valido.any():
            gan = np.where(valido, mx[np.maximum(tope, 0)] - p_q, -np.inf)
            k = int(np.argmax(gan))
            mejor = (gan[k].item(), int(quitar[k]), int(orden[idx_max[tope[k]]]))
    if len(dentro):
        i = int(dentro[np.argmin(inst.p[dentro])])
        d = -inst.p[i].item()
        if mejor is None or d > mejor[0]:
            mejor = (d, i, -1)
    return mejor


def _aplicar(x, i, j):
    if i >= 0:
        x[i] = False
    if j >= 0:
        x[j] = True


# ---------------------------------------------------------------------------
#  Busqueda tabu
# ---------------------------------------------------------------------------
def tabu(inst, semilla=0, tenencia=10, reinicio=100, max_iter=1000, max_sin_mejora=300,
         limite=None):
    """
    S <- BL(voraz).  En cada iteracion se toma el mejor vecino ADMISIBLE (aunque empeore):
      * atributo tabu = el objeto que se movio: si se saco, no puede volver a entrar; si se
        metio, no puede salir.  La duracion se sortea en [tenencia/2, 3*tenencia/2] para
        no recorrer los mismos ciclos (con duracion fija la busqueda deshacia sus pasos
        en el mismo orden apenas vencian los tabu);
      * criterio de aspiracion: un movimiento tabu se permite si lleva a una solucion mejor
        que la mejor encontrada;
      * intensificacion: cada `reinicio` iteraciones sin mejorar se vuelve a la mejor
        solucion conocida (memoria de plazo intermedio, muy simple).
    Para cuando se cumplen max_iter iteraciones, max_sin_mejora sin mejorar la mejor,
    se alcanza la cota de Dantzig o se acaba el tiempo.
    """
    rng = random.Random(semilla)
    fin = m._fin(limite)
    cota = m.cota_dantzig(inst)
    x, _ = m.busqueda_local(inst, m.voraz(inst), fin=fin)
    fx = m.valor(inst, x)
    mejor, fmejor = x.copy(), fx
    hasta = np.zeros(inst.n, dtype=np.int64)      # objeto i es tabu mientras it < hasta[i]
    todos = np.ones(inst.n, dtype=bool)
    ordw = np.argsort(inst.w, kind="stable")
    lo, hi = max(1, tenencia // 2), max(1, 3 * tenencia // 2)
    it = sin_mejora = 0
    while it < max_iter and sin_mejora < max_sin_mejora and fmejor < cota and _ahora() <= fin:
        it += 1
        mov = mejor_movimiento(inst, x, todos, ordw)
        if mov is None:
            break
        if fx + mov[0] <= fmejor:                 # no aplica la aspiracion: solo no-tabu
            mov = mejor_movimiento(inst, x, hasta <= it, ordw)
            if mov is None:
                break
        d, i, j = mov
        _aplicar(x, i, j)
        fx += d
        for k in (i, j):
            if k >= 0:
                hasta[k] = it + rng.randint(lo, hi)
        if fx > fmejor:
            mejor, fmejor, sin_mejora = x.copy(), fx, 0
        else:
            sin_mejora += 1
            if sin_mejora % reinicio == 0:
                x, fx = mejor.copy(), fmejor
    return mejor, {"iter": it}


# ---------------------------------------------------------------------------
#  Recocido simulado
# ---------------------------------------------------------------------------
class _Conjuntos:
    """Objetos dentro / fuera con seleccion aleatoria y actualizacion en O(1)."""

    def __init__(self, x):
        self.dentro = [int(i) for i in np.flatnonzero(x)]
        self.fuera = [int(i) for i in np.flatnonzero(~x)]
        self.pos = {}
        for k, i in enumerate(self.dentro):
            self.pos[i] = k
        for k, i in enumerate(self.fuera):
            self.pos[i] = k

    @staticmethod
    def _sacar(lista, pos, i):
        k = pos[i]
        ult = lista[-1]
        lista[k] = ult
        pos[ult] = k
        lista.pop()

    def meter(self, i):
        self._sacar(self.fuera, self.pos, i)
        self.pos[i] = len(self.dentro)
        self.dentro.append(i)

    def quitar(self, i):
        self._sacar(self.dentro, self.pos, i)
        self.pos[i] = len(self.fuera)
        self.fuera.append(i)


def recocido(inst, semilla=0, alfa=0.95, prob0=0.3, mov_por_temp=None, t_final=1e-3,
             prob_quitar=0.1, limite=None):
    """
    Recocido simulado (Clase 5), maximizando f:
      vecino aleatorio: se escoge j fuera de la mochila al azar y se mete; si no cabe,
         se sacan objetos de la mochila al azar hasta que quepa (un movimiento "1 entra,
         k salen", que permite tanto subir como bajar la cantidad de objetos).
      se acepta con probabilidad 1 si no empeora y exp(delta / T) si empeora.
      T0: tal que un empeoramiento promedio se acepte con probabilidad prob0.
      Enfriamiento geometrico T <- alfa * T cada mov_por_temp movimientos propuestos
      (por defecto max(1000, n)), hasta T < t_final * T0.
    Parte del voraz.  Devuelve la mejor solucion vista.
    """
    rng = random.Random(semilla)
    fin = m._fin(limite)
    p, w = inst.p.tolist(), inst.w.tolist()
    x = m.voraz(inst)
    cj = _Conjuntos(x)
    xl = x.tolist()
    r = inst.c - m.peso(inst, x)
    fx = m.valor(inst, x)
    mejor, fmejor = x.copy(), fx
    L = mov_por_temp or max(1000, inst.n)

    def proponer():
        """
        Vecino aleatorio de la vecindad de la busqueda local mas QUITAR:
          con prob. prob_quitar: QUITAR(i) con i al azar dentro;
          si no: j al azar fuera, i al azar dentro; AGREGAR(j) si cabe,
          INTERCAMBIO(i, j) si cabe el intercambio, y si no, no hay movimiento.
        """
        if cj.dentro and (not cj.fuera or rng.random() < prob_quitar):
            i = cj.dentro[rng.randrange(len(cj.dentro))]
            return -p[i], i, -1
        if not cj.fuera:
            return None
        j = cj.fuera[rng.randrange(len(cj.fuera))]
        if w[j] <= r:
            return p[j], -1, j
        if not cj.dentro:
            return None
        i = cj.dentro[rng.randrange(len(cj.dentro))]
        if w[j] <= r + w[i]:
            return p[j] - p[i], i, j
        return None

    # temperatura inicial a partir de una muestra de movimientos que empeoran
    peores = []
    for _ in range(500):
        mv = proponer()
        if mv is not None and mv[0] < 0:
            peores.append(-mv[0])
    T0 = (sum(peores) / len(peores)) / -math.log(prob0) if peores else 1.0
    T = T0
    movs = acept = 0
    while T > t_final * T0 and _ahora() <= fin:
        for _ in range(L):
            mv = proponer()
            movs += 1
            if mv is None:
                continue
            d, i, j = mv
            if d >= 0 or rng.random() < math.exp(d / T):
                acept += 1
                if i >= 0:
                    xl[i] = False
                    cj.quitar(i)
                    r += w[i]
                if j >= 0:
                    xl[j] = True
                    cj.meter(j)
                    r -= w[j]
                fx += d
                if fx > fmejor:
                    fmejor = fx
                    mejor = np.array(xl, dtype=bool)
        T *= alfa
    return mejor, {"movimientos": movs, "aceptados": acept, "T0": T0}


# ---------------------------------------------------------------------------
#  GRASP
# ---------------------------------------------------------------------------
class _Fenwick:
    """Arbol de Fenwick para contar y escoger el k-esimo objeto vivo en O(log n)."""

    def __init__(self, n):
        self.n = n
        self.t = [0] * (n + 1)
        for i in range(1, n + 1):          # todos vivos al inicio
            self.t[i] += 1
            j = i + (i & -i)
            if j <= n:
                self.t[j] += self.t[i]
        self.log = 1 << (n.bit_length() - 1) if n else 0

    def quitar(self, i):                    # i en base 0
        i += 1
        while i <= self.n:
            self.t[i] -= 1
            i += i & -i

    def prefijo(self, i):                   # vivos en posiciones [0, i)
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i
        return s

    def kesimo(self, k):                    # posicion (base 0) del k-esimo vivo, k en base 0
        pos, paso = 0, self.log
        k += 1
        while paso:
            sig = pos + paso
            if sig <= self.n and self.t[sig] < k:
                pos = sig
                k -= self.t[sig]
            paso >>= 1
        return pos


def construccion_aleatorizada(inst, alfa, rng, orden=None, raz=None):
    """
    Construccion voraz aleatorizada (Clase 5).  Candidatos = objetos fuera que caben.
    Costo de un candidato = razon p/w.  En cada paso:
        RCL = { e candidato : razon(e) >= rmin + alfa * (rmax - rmin) }
    y se escoge un elemento de la RCL uniformemente.  alfa = 1 es el voraz, alfa = 0 es
    completamente aleatorio.
    Truco: como la capacidad solo baja, un objeto que hoy no cabe no cabra nunca, y se
    puede descartar para siempre.  Con los objetos ordenados por razon, los candidatos que
    quedan forman una lista ordenada; rmax y rmin estan en sus extremos y la RCL es un
    prefijo.  Un arbol de Fenwick permite escoger un elemento al azar del prefijo.
    """
    if orden is None:
        raz = m.razon(inst)
        orden = np.argsort(-raz, kind="stable")
    raz_ord = (-raz[orden]).tolist()      # creciente: sirve para busqueda binaria
    w_ord = inst.w[orden].tolist()
    n = inst.n
    fw = _Fenwick(n)
    vivo = [True] * n
    ini, fin_ = 0, n - 1
    cap = inst.c
    x = np.zeros(n, dtype=bool)

    def matar(k):
        vivo[k] = False
        fw.quitar(k)

    vivos = n
    while vivos:
        while ini <= fin_ and (not vivo[ini] or w_ord[ini] > cap):
            if vivo[ini]:
                matar(ini)
                vivos -= 1
            ini += 1
        while fin_ >= ini and (not vivo[fin_] or w_ord[fin_] > cap):
            if vivo[fin_]:
                matar(fin_)
                vivos -= 1
            fin_ -= 1
        if ini > fin_:
            break
        rmax, rmin = -raz_ord[ini], -raz_ord[fin_]
        umbral = rmin + alfa * (rmax - rmin)
        tope = bisect.bisect_right(raz_ord, -umbral, ini, fin_ + 1)   # RCL = [ini, tope)
        tope = max(tope, ini + 1)          # (redondeo: el mejor candidato siempre esta)
        base = fw.prefijo(ini)
        cuantos = fw.prefijo(tope) - base
        k = fw.kesimo(base + rng.randrange(cuantos))
        if w_ord[k] <= cap:
            x[orden[k]] = True
            cap -= w_ord[k]
        matar(k)                       # escogido o (si no cabe) descartado para siempre
        vivos -= 1
    return x


def grasp(inst, semilla=0, alfa=0.97, max_iter=30, max_sin_mejora=10, reactivo=False,
          alfas=(0.9, 0.95, 0.97, 0.99), limite=None):
    """
    GRASP (Clase 5): repetir construccion voraz aleatorizada + busqueda local, quedarse
    con la mejor.  Version reactiva: alfa se escoge de `alfas` con probabilidad
    proporcional a q_k = A_k / f(S*), con A_k el promedio de las soluciones obtenidas con
    alfa_k (la formula de la clase, invertida porque aqui se maximiza).
    """
    rng = random.Random(semilla)
    fin = m._fin(limite)
    cota = m.cota_dantzig(inst)
    raz = m.razon(inst)
    orden = np.argsort(-raz, kind="stable")
    mejor, fmejor = None, -1
    suma = [0.0] * len(alfas)
    veces = [0] * len(alfas)
    it = sin_mejora = 0
    while it < max_iter and sin_mejora < max_sin_mejora and fmejor < cota and _ahora() <= fin:
        it += 1
        if reactivo:
            if 0 in veces:
                a = veces.index(0)
            else:
                q = [suma[k] / veces[k] / fmejor for k in range(len(alfas))]
                a = rng.choices(range(len(alfas)), weights=q)[0]
            al = alfas[a]
        else:
            al = alfa
        x = construccion_aleatorizada(inst, al, rng, orden, raz)
        x, _ = m.busqueda_local(inst, x, fin=fin)
        f = m.valor(inst, x)
        if reactivo:
            suma[a] += f
            veces[a] += 1
        if f > fmejor:
            mejor, fmejor, sin_mejora = x, f, 0
        else:
            sin_mejora += 1
    if mejor is None:                    # ni una iteracion (tiempo agotado): voraz
        mejor = m.voraz(inst)
    return mejor, {"iter": it}


# ---------------------------------------------------------------------------
#  Algoritmo genetico
# ---------------------------------------------------------------------------
class _Reparador:
    """
    Reparacion voraz (Chu y Beasley, 1998).  Con los objetos en orden de razon decreciente:
      1) si el individuo no cabe, se quitan los de peor razon hasta que quepa;
      2) se agregan, en orden de razon, los objetos que todavia quepan.
    Todo vectorizado con NumPy (el paso 2 se repite mientras quepa alguno).
    """

    def __init__(self, inst):
        self.inst = inst
        self.orden = np.argsort(-m.razon(inst), kind="stable")
        self.w_ord = inst.w[self.orden]

    def __call__(self, x):
        c = self.inst.c
        xs = x[self.orden]
        cw = np.cumsum(np.where(xs, self.w_ord, 0))
        if cw[-1] > c:
            xs &= cw <= c                       # se quedan los de mejor razon que caben
        r = c - int(np.where(xs, self.w_ord, 0).sum())
        while True:
            cand = ~xs & (self.w_ord <= r)
            if not cand.any():
                break
            idx = np.flatnonzero(cand)
            cs = np.cumsum(self.w_ord[idx])
            k = int(np.searchsorted(cs, r, side="right"))
            if k == 0:                          # el primero no cabe junto: se mete solo
                k = 1
            xs[idx[:k]] = True
            r -= int(cs[k - 1])
        y = np.empty_like(x)
        y[self.orden] = xs
        return y


def genetico(inst, semilla=0, tam_pob=50, prob_cruce=0.9, cruce="uniforme", prob_mut=None,
             elite=2, max_gen=300, max_sin_mejora=60, limite=None):
    """
    Algoritmo genetico (Clases 6 y 7).
      genotipo: cadena binaria de largo n;  fenotipo: el conjunto de objetos de la mochila.
      Todo genotipo se repara (ver _Reparador) antes de evaluarse, asi que el fenotipo
      siempre es factible y la aptitud es directamente f(x) = suma de valores.
      Poblacion inicial: cadenas aleatorias reparadas.  Cada individuo usa su propia
      P[bit = 1] = u * c / suma(w), con u ~ U(0, 1): con u chico casi todo lo pone la
      reparacion (parecido al voraz) y con u grande casi todo es azar; asi la poblacion
      empieza diversa pero no tan mala (con u = 1 para todos, en instancias grandes la
      mochila se llenaba de objetos al azar y la reparacion ya no podia meter nada).
      Seleccion: torneo binario.  Cruce: uniforme o de un punto, con prob. prob_cruce.
      Mutacion: cada bit cambia con prob. prob_mut (por defecto 2/n).
      Reemplazo generacional con elitismo (los `elite` mejores pasan directo).
      Para: max_gen generaciones, max_sin_mejora sin mejorar, cota de Dantzig o tiempo.
    """
    rng = np.random.default_rng(semilla)
    fin = m._fin(limite)
    n = inst.n
    pm = 2.0 / n if prob_mut is None else prob_mut
    cota = m.cota_dantzig(inst)
    rep = _Reparador(inst)
    q = min(1.0, inst.c / float(inst.w.sum()))
    pob = [rep(rng.random(n) < q * rng.random()) for _ in range(tam_pob)]
    apt = np.array([m.valor(inst, x) for x in pob], dtype=float)
    b = int(np.argmax(apt))
    mejor, fmejor = pob[b].copy(), apt[b]
    gen = sin_mejora = 0

    def torneo():
        a, bb = rng.integers(tam_pob, size=2)
        return pob[a] if apt[a] >= apt[bb] else pob[bb]

    while gen < max_gen and sin_mejora < max_sin_mejora and fmejor < cota and _ahora() <= fin:
        gen += 1
        orden = np.argsort(-apt)
        nueva = [pob[k] for k in orden[:elite]]
        while len(nueva) < tam_pob:
            p1, p2 = torneo(), torneo()
            if rng.random() < prob_cruce:
                if cruce == "uniforme":
                    msk = rng.random(n) < 0.5
                    h1, h2 = np.where(msk, p1, p2), np.where(msk, p2, p1)
                else:                                         # un punto
                    pc = int(rng.integers(1, n))
                    h1 = np.concatenate([p1[:pc], p2[pc:]])
                    h2 = np.concatenate([p2[:pc], p1[pc:]])
            else:
                h1, h2 = p1.copy(), p2.copy()
            for h in (h1, h2):
                k = rng.binomial(n, pm)                       # cuantos bits mutan
                if k:
                    pos = rng.integers(n, size=k)
                    h[pos] = ~h[pos]
                nueva.append(rep(h))
        pob = nueva[:tam_pob]
        apt = np.array([m.valor(inst, x) for x in pob], dtype=float)
        b = int(np.argmax(apt))
        if apt[b] > fmejor:
            mejor, fmejor, sin_mejora = pob[b].copy(), apt[b], 0
        else:
            sin_mejora += 1
    return mejor, {"generaciones": gen}
