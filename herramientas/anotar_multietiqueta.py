# Genera las versiones multietiqueta del dataset y de la generalización (Fase 3) según guia_anotacion_multietiqueta.md.
# No usa ningún modelo: las etiquetas adicionales se fijan a mano en REETIQUETADAS, con su justificación.
# Conserva la partición de la v1 (mismo train_test_split estratificado 70/15/15, semilla 42, que SoporTIC.py).
# Uso: python herramientas/anotar_multietiqueta.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42

# texto exacto -> (intenciones finales, justificación). Vacío: aplicando la guía, ninguna frase describe dos problemas.
REETIQUETADAS = {}

# Casos límite revisados que conservan una etiqueta (regla de la guía que se aplica)
CASOS_LIMITE = {
    "se me olvido la clave y mañana entrego en moodle": "R3: «mañana entrego en moodle» es un plazo, no un problema de entrega.",
    "se fue el internet en medio del quiz, que hago": "R2: la caída de internet es la causa; lo que se pide es qué hacer con el cuestionario.",
    "le di olvide mi contraseña y no me llega ningun correo": "R2: el correo de recuperación es parte del mismo proceso de restablecimiento.",
    "mañana entrego el modelo y colab no me da gpu": "R3: plazo como contexto; el problema es la GPU de Colab.",
    "el computador de la sala no tiene internet": "R4: la definición de salas_computo incluye equipos de la sala sin internet.",
    "en el correo de la u me dice contraseña incorrecta y no la he cambiado": "R2/R4: el usuario sabe su contraseña; es acceso al correo (definición de la Fase 1).",
    "moodle me dice contraseña incorrecta pero estoy segura que es esa": "R2/R4: el usuario sabe su contraseña; es acceso a Moodle.",
    "la plataforma academica me pide una contraseña distinta a la del correo": "R2: el correo aparece como comparación; el problema es el ingreso a la plataforma.",
    "el moodle me rebota al login aunque pongo bien la clave": "R2/R4: acceso a Moodle con contraseña conocida (generalización).",
}

ds = pd.read_csv(os.path.join(RAIZ, "dataset_soporte.csv")).reset_index().rename(columns={"index": "id"})
etiquetas = sorted(ds["intencion"].unique())
ds["label"] = ds["intencion"].map({e: i for i, e in enumerate(etiquetas)})
tr, tmp = train_test_split(ds, test_size=0.30, stratify=ds["label"], random_state=SEED)
va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=SEED)
ds["particion"] = ""
ds.loc[tr.index, "particion"], ds.loc[va.index, "particion"], ds.loc[te.index, "particion"] = "train", "val", "test"

gen = pd.read_csv(os.path.join(RAIZ, "frases_generalizacion.csv")).reset_index().rename(columns={"index": "id"})
for d in (ds, gen):
    d["intenciones"] = [REETIQUETADAS.get(t, (i, ""))[0] for t, i in zip(d["texto"], d["intencion"])]
assert all(c in set(ds["texto"]) | set(gen["texto"]) for c in CASOS_LIMITE), "caso límite inexistente"

ds[["id", "texto", "intenciones", "origen", "particion"]].to_csv(os.path.join(RAIZ, "dataset_soporte_multietiqueta.csv"), index=False)
gen[["id", "texto", "intenciones", "origen"]].to_csv(os.path.join(RAIZ, "frases_generalizacion_multietiqueta.csv"), index=False)
print("particiones:", ds["particion"].value_counts().to_dict())
print("reetiquetadas:", len(REETIQUETADAS))
print("con 2+ intenciones: dataset", int(ds["intenciones"].str.contains(r"\|").sum()), "| generalización",
      int(gen["intenciones"].str.contains(r"\|").sum()))
