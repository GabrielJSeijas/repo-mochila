# Proyecto Mochila 0/1 — CI5652 Diseño de Algoritmos II (Etapa 2)

Ángel Valero (18-10436), Alejandra Mármol (20-10408), Gabriel Seijas (19-00036).

## Archivos

| Archivo | Qué tiene |
|---|---|
| `mochila.py` | Lectura de instancias y algoritmos de la Etapa 1 (PD, voraz, BL, ILS) |
| `etapa2.py` | BL desde aleatorias, búsqueda tabú, recocido simulado, GRASP, genético |
| `generar.py` | Genera las 16 instancias del grupo G (ya están en `inst/generadas`) |
| `experimentos2.py` | Corre todo y escribe `resultados2.csv` / `parametros2.csv` |
| `tablas_e2.py` | Arma las tablas LaTeX del informe (`tablas/`) a partir de los CSV |
| `inst/` | Instancias: B (Ortega), H (Pisinger, solo las 2 primeras de cada archivo), G (generadas, de 10 000 a 100 000 objetos) |

## Correr en la nube con GitHub Actions (gratis)

1. Crear un repo **privado** en GitHub y subir esta carpeta tal cual (incluida `.github/`).
2. En el repo: pestaña **Actions** → workflow **experimentos** → **Run workflow**.
3. Se lanza una máquina por instancia (40 para la corrida principal y 6 para el estudio de parámetros, hasta 20 a la vez). Al final, el job `tablas` une todo.
4. Cuando termine (aprox. 3 a 4 h, por las 4 instancias de 100 000 objetos), entrar a la corrida y bajar el artifact **resultados** (zip con `resultados2.csv`, `parametros2.csv`, `tablas/` y `res/`).

Máquinas: `ubuntu-latest` de GitHub, cada instancia en su propia máquina y en un solo proceso.

## Correr en una máquina propia

```bash
pip install -r requirements.txt
python3 experimentos2.py principal 2      # 2 procesos en paralelo
python3 experimentos2.py parametros 2 knapPI_1_10000_1000_1,knapPI_2_2000_1000_1,knapPI_14_10000_1000_1,knapPI_16_10000_1000_1,gen_3_10000_100000,gen_5_10000_100000
python3 tablas_e2.py
```

Si se interrumpe, se vuelve a correr el mismo comando y retoma: cada instancia terminada queda guardada en `res/<modo>/`.
