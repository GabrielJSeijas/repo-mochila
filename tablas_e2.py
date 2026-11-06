"""Genera las tablas LaTeX del informe (carpeta tablas/) a partir de resultados2.csv y parametros2.csv."""
import csv
import json
import os
import random
import statistics as st
import sys

import etapa2 as e
import experimentos2 as X
import mochila as m

os.makedirs("tablas", exist_ok=True)
R = list(csv.DictReader(open("resultados2.csv")))
P = list(csv.DictReader(open("parametros2.csv"))) if os.path.exists("parametros2.csv") else []
for r in R + P:
    r["vals"] = [int(v) for v in r["valores"].split()] if r["valores"] else []
    r["ts"] = [float(t) for t in r["tiempos"].split()] if r["tiempos"] else []

INST = X.lista_instancias()
NOMBRE = dict(INST)
GRUPOS = ("B", "H", "G")
ALGS = ["voraz", "bl_voraz", "bl_aleat", "ils", "tabu", "recocido", "grasp", "genetico"]
ETQ = {"pd": "PD", "voraz": "Voraz", "bl_voraz": "BL (voraz)", "bl_aleat": "BL (aleat.)",
       "ils": "ILS", "tabu": "Tabú", "recocido": "Recocido", "grasp": "GRASP", "genetico": "Genético"}
CORTA = {"voraz": "Vor.", "bl_voraz": "BL", "bl_aleat": "BLa", "ils": "ILS", "tabu": "Tabú",
         "recocido": "RS", "grasp": "GR", "genetico": "AG"}
ALEAT = ["bl_aleat", "ils", "tabu", "recocido", "grasp", "genetico"]

por = {}
for r in R:
    por[(r["instancia"], r["algoritmo"])] = r
hechas = [(g, nm) for g, nm in INST if (nm, "voraz") in por]

# ---------------- referencia de cada instancia
ref, ref_info = {}, {}
for g, nm in hechas:
    if g in ("B", "H"):
        I = X.cargar(g, nm)
        ref[nm] = int(I.optimo)
        ref_info[nm] = "publicado"
    else:
        o = por[(nm, "ortools")]
        mejor = max([v for r in R + P if r["instancia"] == nm and r["algoritmo"] != "cota_dantzig"
                     for v in r["vals"]])
        if o["variante"] == "optimo":
            assert mejor <= o["vals"][0]
            ref[nm], ref_info[nm] = o["vals"][0], "OR-Tools (óptimo)"
        else:
            vo = o["vals"][0] if o["vals"] else -1          # sin valor si OR-Tools se quedo sin memoria
            ref[nm] = max(mejor, vo)
            ref_info[nm] = "mejor conocido (%s)" % ("OR-Tools" if vo >= mejor else "nuestro")
    if (nm, "pd") in por and por[(nm, "pd")]["variante"] in ("ok", "solo_valor"):
        assert por[(nm, "pd")]["vals"][0] == ref[nm], nm


def gaps(nm, r):
    return [100.0 * (ref[nm] - v) / ref[nm] for v in r["vals"]]


def fg(v, neg=False):
    """formato de un gap en %"""
    if v < 5e-7:
        return "0"
    if v >= 1:
        return "%.2f" % v
    if v >= 0.01:
        return "%.3f" % v
    if v >= 0.0001:
        return "%.4f" % v
    return "%.1e" % v


def ft(t):
    """tiempo en segundos, con unidad"""
    if t < 1e-3:
        return "%.2f\\,ms" % (1000 * t)
    if t < 1:
        return "%.0f\\,ms" % (1000 * t) if t >= 0.1 else "%.1f\\,ms" % (1000 * t)
    if t < 100:
        return "%.1f\\,s" % t
    return "%.0f\\,s" % t


def esc(s):
    return s.replace("_", "\\_")


def nombre_corto(nm):
    return "\\texttt{%s}" % esc(nm.replace("_1000_", "\\_").replace("knapPI_", "kp\\_") if False else nm)


S = {}       # cifras sueltas para el texto
out = {}

# ---------------- tabla por instancia: gap promedio de cada algoritmo
for g in GRUPOS:
    filas = []
    for gg, nm in hechas:
        if gg != g:
            continue
        pd = por[(nm, "pd")]
        if pd["variante"] in ("ok", "solo_valor"):
            pdtxt = ft(pd["ts"][0])
        else:
            pdtxt = "\\textit{mem.}" if pd["variante"] == "memoria" else "\\textit{t.o.}"
        medias = [st.mean(gaps(nm, por[(nm, a)])) for a in ALGS]
        mejor = min(medias)
        celdas = []
        for a, v in zip(ALGS, medias):
            txt = fg(v)
            celdas.append("\\textbf{%s}" % txt if abs(v - mejor) < 1e-12 else txt)
        n = por[(nm, "voraz")]["n"]
        filas.append("\\texttt{%s} & %s & %s & %s \\\\ \\hline" % (
            esc(nm), n, pdtxt, " & ".join(celdas)))
    out["inst_" + g] = "\n".join(filas)

# ---------------- tabla por instancia: tiempo promedio
for g in GRUPOS:
    filas = []
    for gg, nm in hechas:
        if gg != g:
            continue
        celdas = [ft(st.mean(por[(nm, a)]["ts"])) for a in ALGS]
        filas.append("\\texttt{%s} & %s \\\\ \\hline" % (esc(nm), " & ".join(celdas)))
    out["tiempo_" + g] = "\n".join(filas)

# ---------------- resumen global por grupo
def resumen_alg(a, inst):
    if a == "pd":
        ok = [nm for nm in inst if por[(nm, "pd")]["variante"] in ("ok", "solo_valor")]
        return len(ok), None, None, None, None
    medias = [st.mean(gaps(nm, por[(nm, a)])) for nm in inst]
    mejores = [min(gaps(nm, por[(nm, a)])) for nm in inst]
    peores = [max(gaps(nm, por[(nm, a)])) for nm in inst]
    alcanz = sum(1 for nm in inst if min(gaps(nm, por[(nm, a)])) < 1e-9)
    corr = sum(sum(1 for x in gaps(nm, por[(nm, a)]) if x < 1e-9) for nm in inst)
    return alcanz, st.mean(medias), st.mean(mejores), st.mean(peores), corr


filas = []
for a in ["pd"] + ALGS:
    celdas = []
    for g in GRUPOS:
        inst = [nm for gg, nm in hechas if gg == g]
        if not inst:
            celdas += ["", ""]
            continue
        k, med, _, _, _ = resumen_alg(a, inst)
        S.setdefault(a, {})[g] = dict(alcanz=k, gap=med, n=len(inst))
        if a == "pd":
            celdas += ["%d/%d" % (k, len(inst)), "0" if k == len(inst) else "---"]
        else:
            celdas += ["%d" % k, fg(med)]
    filas.append("%s & %s \\\\ \\hline" % (ETQ[a], " & ".join(celdas)))
out["resumen"] = "\n".join(filas)

# ---------------- variabilidad de los aleatorios (mejor / media / peor corrida) y corridas en la ref
filas = []
for a in ALEAT:
    celdas = []
    for g in GRUPOS:
        inst = [nm for gg, nm in hechas if gg == g]
        k, med, mej, peo, corr = resumen_alg(a, inst)
        S[a][g].update(mejor=mej, peor=peo, corridas=corr)
        celdas += [fg(mej), fg(med), fg(peo), "%d" % corr]
    filas.append("%s & %s \\\\ \\hline" % (ETQ[a], " & ".join(celdas)))
out["variab"] = "\n".join(filas)

# ---------------- tiempos medios por grupo
filas = []
for a in ["pd"] + ALGS:
    celdas = []
    for g in GRUPOS:
        inst = [nm for gg, nm in hechas if gg == g]
        if a == "pd":
            ok = [por[(nm, "pd")]["ts"][0] for nm in inst if por[(nm, "pd")]["variante"] in ("ok", "solo_valor")]
            celdas.append(ft(st.mean(ok)) if ok else "sin solución")
            S["pd"][g]["t"] = st.mean(ok) if ok else None
            continue
        ts = [st.mean(por[(nm, a)]["ts"]) for nm in inst]
        tmax = max(max(por[(nm, a)]["ts"]) for nm in inst)
        S[a][g].update(t=st.mean(ts), tmax=tmax)
        celdas.append("%s {\\scriptsize(máx. %s)}" % (ft(st.mean(ts)), ft(tmax)))
    filas.append("%s & %s \\\\ \\hline" % (ETQ[a], " & ".join(celdas)))
out["tiempos"] = "\n".join(filas)

# ---------------- grupo G: referencia
filas = []
for gg, nm in hechas:
    if gg != "G":
        continue
    o = por[(nm, "ortools")]
    U = por[(nm, "cota_dantzig")]["vals"][0]
    filas.append("\\texttt{%s} & %s & %s & %s & %s & %s & %s \\\\ \\hline" % (
        esc(nm), por[(nm, "voraz")]["n"], "{:,}".format(int(por[(nm, "voraz")]["c"])).replace(",", "\\,"),
        "{:,}".format(ref[nm]).replace(",", "\\,"),
        {"optimo": "sí", "no_demostrado": "no"}.get(o["variante"], "no (mem.)"), ft(o["ts"][0]),
        fg(100.0 * (U - ref[nm]) / U)))
out["refG"] = "\n".join(filas)
S["G_ortools_opt"] = sum(1 for gg, nm in hechas if gg == "G" and por[(nm, "ortools")]["variante"] == "optimo")
S["G_ortools"] = {nm: dict(estado=por[(nm, "ortools")]["variante"], t=por[(nm, "ortools")]["ts"][0],
                           ortools=(por[(nm, "ortools")]["vals"] or [None])[0],
                           mejor_nuestro=max(v for a in ALEAT + ["voraz", "bl_voraz"] for v in por[(nm, a)]["vals"]),
                           quien=[a for a in ALEAT + ["bl_voraz"] if max(por[(nm, a)]["vals"]) == ref[nm]])
                  for gg, nm in hechas if gg == "G"}
S["G_ref_nuestra"] = [nm for gg, nm in hechas if gg == "G" and ref_info[nm].endswith("(nuestro)")]

# ---------------- BL desde aleatorio vs desde el voraz
filas = []
for g in GRUPOS:
    inst = [nm for gg, nm in hechas if gg == g]
    g_ini, g_fin, g_vor, g_blv, pasos = [], [], [], [], []
    for nm in inst:
        I = X.cargar(g, nm)
        ini = [m.valor(I, e.solucion_aleatoria(I, random.Random(s))) for s in X.SEMILLAS]
        g_ini.append(st.mean(100.0 * (ref[nm] - v) / ref[nm] for v in ini))
        g_fin.append(st.mean(gaps(nm, por[(nm, "bl_aleat")])))
        g_vor.append(st.mean(gaps(nm, por[(nm, "voraz")])))
        g_blv.append(st.mean(gaps(nm, por[(nm, "bl_voraz")])))
    filas.append("%s & %s & %s & %s & %s & %s \\\\ \\hline" % (
        g, fg(st.mean(g_ini)), fg(st.mean(g_fin)), fg(st.mean(g_vor)), fg(st.mean(g_blv)),
        ft(st.mean(st.mean(por[(nm, "bl_aleat")]["ts"]) for nm in inst))))
    S.setdefault("bla", {})[g] = dict(ini=st.mean(g_ini), fin=st.mean(g_fin), vor=st.mean(g_vor),
                                       blv=st.mean(g_blv))
out["blaleat"] = "\n".join(filas)

# ---------------- estudio de parametros
if P:
    pinst = sorted(set(r["instancia"] for r in P), key=lambda nm: [x[1] for x in INST].index(nm))
    S["param_inst"] = pinst
    filas = []
    for a, vs in X.VARIANTES.items():
        for k, (etq, _) in enumerate(vs):
            rows = [por[(nm, a)] if k == 0 else
                    next(r for r in P if r["instancia"] == nm and r["algoritmo"] == a and r["variante"] == etq)
                    for nm in pinst]
            gm = st.mean(st.mean(gaps(r["instancia"], r)) for r in rows)
            gp = st.mean(max(gaps(r["instancia"], r)) for r in rows)
            hits = sum(1 for r in rows for x in gaps(r["instancia"], r) if x < 1e-9)
            tm = st.mean(st.mean(r["ts"]) for r in rows)
            S.setdefault("param", {}).setdefault(a, {})[etq] = dict(gap=gm, peor=gp, hits=hits, t=tm)
            filas.append("%s & %s%s & %s & %s & %d/%d & %s \\\\ %s" % (
                ETQ[a] if k == 0 else "", etq.replace("a=", "$\\alpha=$").replace("t=", "$t=$").replace(
                    "pm=10/n", "$p_m=10/n$").replace("N=", "$N=$"),
                " (defecto)" if k == 0 else "", fg(gm), fg(gp), hits, 10 * len(pinst), ft(tm),
                "\\hline" if k == len(vs) - 1 else "\\cline{2-6}"))
    out["param"] = "\n".join(filas)

for k, v in out.items():
    open(os.path.join("tablas", k + ".tex"), "w").write(v + "\n")
json.dump(S, open("cifras.json", "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(S, indent=1, ensure_ascii=False, default=str)[:6000])
