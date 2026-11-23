# Analiza tareas_evaluacion.csv (evaluación humana de SoporTIC), que llenan los dos autores por separado.
# Calcula, por evaluador, el porcentaje de tareas resueltas y la calificación media (1 a 5), y el acuerdo entre evaluadores:
# acuerdo observado y kappa de Cohen en «resuelta» (sí/no); diferencia media absoluta y kappa ponderada cuadrática en la
# calificación. Solo usa las tareas que ambos evaluadores calificaron.
# Uso: python herramientas/analizar_evaluacion_humana.py
import os, sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA = os.path.join(RAIZ, "tareas_evaluacion.csv")


def a_binario(serie):
    s = serie.astype(str).str.strip().str.lower().str.replace("í", "i")
    return s.map({"si": 1, "sí": 1, "s": 1, "1": 1, "no": 0, "n": 0, "0": 0})


def kappa(a, b, categorias, pesos=None):
    """Kappa de Cohen; con pesos='cuadratica', kappa ponderada cuadrática."""
    k = len(categorias)
    idx = {c: i for i, c in enumerate(categorias)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    n = O.sum()
    if n == 0:
        return float("nan")
    E = np.outer(O.sum(1), O.sum(0)) / n
    i, j = np.indices((k, k))
    W = ((i - j) ** 2 / (k - 1) ** 2) if pesos == "cuadratica" else (i != j).astype(float)
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def main():
    d = pd.read_csv(RUTA, dtype=str)
    r1, r2 = a_binario(d["evaluador1_resuelta_si_no"]), a_binario(d["evaluador2_resuelta_si_no"])
    c1 = pd.to_numeric(d["evaluador1_calificacion_1_5"], errors="coerce")
    c2 = pd.to_numeric(d["evaluador2_calificacion_1_5"], errors="coerce")
    if r1.isna().all() and r2.isna().all() and c1.isna().all() and c2.isna().all():
        print("Evaluación humana: tareas_evaluacion.csv aún no tiene calificaciones (la llenan los autores).")
        return
    for nombre, r, c in [("Evaluador 1", r1, c1), ("Evaluador 2", r2, c2)]:
        print(f"{nombre}: tareas calificadas {int(r.notna().sum())} | resueltas {100 * r.mean():.1f} % | "
              f"calificación media {c.mean():.2f} (DE {c.std():.2f})")
    ambos_r = r1.notna() & r2.notna()
    if ambos_r.any():
        a, b = r1[ambos_r].astype(int), r2[ambos_r].astype(int)
        print(f"Acuerdo en «resuelta» ({int(ambos_r.sum())} tareas): observado {100 * (a == b).mean():.1f} % | "
              f"kappa de Cohen {kappa(a, b, [0, 1]):.3f}")
    ambos_c = c1.notna() & c2.notna()
    if ambos_c.any():
        a, b = c1[ambos_c].astype(int), c2[ambos_c].astype(int)
        print(f"Calificación ({int(ambos_c.sum())} tareas): media conjunta {pd.concat([a, b]).mean():.2f} | "
              f"diferencia media absoluta {np.abs(a - b).mean():.2f} | kappa ponderada cuadrática "
              f"{kappa(a, b, [1, 2, 3, 4, 5], 'cuadratica'):.3f}")
        print("Por tarea (media de los dos evaluadores):")
        print(pd.DataFrame({"tarea": d.loc[ambos_c, "id"], "media": (a + b) / 2}).to_string(index=False))


if __name__ == "__main__":
    main()
