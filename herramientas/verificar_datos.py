# Verificación de los datos de SoporTIC: conteos, cobertura de slots, duplicados, casi duplicados (MiniLM)
# y fuga entre dataset, generalización y conjunto externo. Solo lee; no modifica ningún archivo.
# Uso: python herramientas/verificar_datos.py [carpeta_del_proyecto]
import os, re, sys, unicodedata
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UMBRAL = 0.90
DISPOSITIVO = r"\b(celular|cel|portatil|pc|computador|computadores|equipo|tablet|mac|iphone|laptop)\b"
ERROR = (r"(error|no carga|no abre|no prende|contraseña incorrecta|contrasena incorrecta|pantalla en blanco|pantalla blanca|"
         r"pantalla negra|queda negra|se cierra|se queda|no funciona|no sirve|caido|no me deja|denegado|no valido|no es valido)")


def norm(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def leer_externo(ruta):
    """Lee frase;intencion o frase;intencion1|intencion2; ignora líneas vacías y comentarios (#)."""
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for n, linea in enumerate(f, 1):
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            texto, etiquetas = linea.rsplit(";", 1)
            filas.append({"linea": n, "texto": texto.strip(), "intencion": etiquetas.strip(),
                          "n_intenciones": len(etiquetas.split("|"))})
    return pd.DataFrame(filas, columns=["linea", "texto", "intencion", "n_intenciones"])


ds = pd.read_csv(os.path.join(d, "dataset_soporte.csv"))
gen = pd.read_csv(os.path.join(d, "frases_generalizacion.csv"))
ext = leer_externo(os.path.join(d, "frases_prueba_externa.txt"))
INTENCIONES = sorted(ds["intencion"].unique())

print("== Conteos por intención ==")
ext1 = ext[ext["n_intenciones"] == 1]
tab = pd.DataFrame({"dataset": ds["intencion"].value_counts(), "generalizacion": gen["intencion"].value_counts(),
                    "externo_1": ext1["intencion"].value_counts()}).fillna(0).astype(int)
tab["con_dispositivo"] = ds.groupby("intencion")["texto"].apply(lambda s: s.str.contains(DISPOSITIVO, regex=True).sum())
tab["con_error"] = ds.groupby("intencion")["texto"].apply(lambda s: s.apply(norm).str.contains(ERROR, regex=True).sum())
tab.loc["TOTAL"] = tab.sum()
print(tab.to_string())
print("origen:", ds["origen"].value_counts().to_dict(), gen["origen"].value_counts().to_dict())
print("externo por número de intenciones:", ext["n_intenciones"].value_counts().sort_index().to_dict(), "| total:", len(ext))

print("\n== Etiquetas del conjunto externo ==")
malas = [(r.linea, r.texto, e) for r in ext.itertuples() for e in r.intencion.split("|") if e not in INTENCIONES]
print("etiquetas inexistentes:", len(malas))
for m in malas:
    print("  ", m)
rep = [(r.linea, r.texto) for r in ext.itertuples() if len(set(r.intencion.split("|"))) != r.n_intenciones]
print("etiquetas repetidas en una línea:", len(rep))

print("\n== Duplicados exactos / normalizados ==")
for nombre, df in [("dataset", ds), ("generalizacion", gen), ("externo", ext)]:
    print(f"{nombre}: exactos={df['texto'].str.lower().str.strip().duplicated().sum()} "
          f"normalizados={df['texto'].apply(norm).duplicated().sum()}")
todo = pd.concat([ds.assign(conj="dataset"), gen.assign(conj="generalizacion"), ext.assign(conj="externo")],
                 ignore_index=True)
todo["n"] = todo["texto"].apply(norm)
for (a, b) in [("dataset", "generalizacion"), ("dataset", "externo"), ("generalizacion", "externo")]:
    exactas = set(todo.loc[todo.conj == a, "texto"].str.lower().str.strip()) & set(todo.loc[todo.conj == b, "texto"].str.lower().str.strip())
    normal = set(todo.loc[todo.conj == a, "n"]) & set(todo.loc[todo.conj == b, "n"])
    print(f"{a} vs {b}: exactas={len(exactas)} normalizadas={len(normal)}")
    for t in sorted(normal):
        print("   =", t)

print("\n== Casi duplicados MiniLM (coseno >= 0.90) ==")
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=dev)
E = m.encode(todo["texto"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
S = (E @ E.T).cpu()
pares = []
for i in range(len(todo)):
    for j in range(i + 1, len(todo)):
        if S[i, j] >= UMBRAL:
            pares.append((round(float(S[i, j]), 3), todo.conj[i], todo.texto[i], todo.intencion[i],
                          todo.conj[j], todo.texto[j], todo.intencion[j]))
for p in sorted(pares, key=lambda p: (p[1], p[4], -p[0])):
    print(p)
combos = pd.Series([" vs ".join(sorted([p[1], p[4]])) for p in pares]).value_counts().to_dict() if pares else {}
print("pares >= 0.90:", len(pares), "| por combinación:", combos)
idd = todo.index[todo.conj == "dataset"]
for c in ["generalizacion", "externo"]:
    ic = todo.index[todo.conj == c]
    if len(ic):
        mx = S[ic][:, idd].max(1).values
        print(f"{c} vs dataset: similitud máxima media={mx.mean():.3f}, máx={mx.max():.3f}")

# ---------- Externo vs aumento del entrenamiento de la v1 (si existe la caché) ----------
ruta_aum = os.path.join(d, "soportic", "datos", "aumento_train.csv")
if os.path.exists(ruta_aum):
    aum = pd.read_csv(ruta_aum)
    E_aum = m.encode(aum["texto"].astype(str).tolist(), convert_to_tensor=True, normalize_embeddings=True)
    ie = todo.index[todo.conj == "externo"]
    S_ea = (E[ie] @ E_aum.T).cpu()
    mx, arg = S_ea.max(dim=1)
    print("\n== Externo vs aumento_train.csv (v1) ==")
    n = 0
    for k, i in enumerate(ie):
        if mx[k] >= UMBRAL:
            n += 1
            print((round(float(mx[k]), 3), todo.texto[i], todo.intencion[i], "| aumento:", aum["texto"].iloc[int(arg[k])],
                   aum["intencion"].iloc[int(arg[k])], aum["tecnica"].iloc[int(arg[k])]))
    print(f"frases del externo con similitud >= {UMBRAL} con el aumento: {n} | máxima: {float(mx.max()):.3f}")
