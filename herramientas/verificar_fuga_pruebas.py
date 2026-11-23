# Comprueba, antes de ejecutarlas, que ninguna frase de la batería (bateria_soporte.json) ni de las conversaciones de la
# memoria (conversaciones_memoria.json) tenga similitud MiniLM >= 0,90 con el entrenamiento original o el aumento de la v1.
# Uso: python herramientas/verificar_fuga_pruebas.py
import os, sys, json, re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RAIZ)
SIM_MAX = 0.90

def normalizar_texto(texto):        # misma normalización que el notebook (sección 4)
    texto = str(texto).lower().strip()
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"(.)\1{2,}", r"\1\1", texto)
    texto = re.sub(r"[^a-záéíóúüñ0-9¿?¡!.,\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()

df = pd.read_csv("dataset_soporte.csv")
etiquetas = sorted(df["intencion"].unique())
df["label"] = df["intencion"].map({e: i for i, e in enumerate(etiquetas)})
train, _ = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
aumento = pd.read_csv("soportic/datos/aumento_train.csv")
referencia = train["texto"].tolist() + aumento["texto"].astype(str).tolist()

bateria = json.load(open("bateria_soporte.json", encoding="utf-8"))["casos"]
conversaciones = json.load(open("conversaciones_memoria.json", encoding="utf-8"))["conversaciones"]
frases = {}
for c in bateria:
    for t in c["turnos"]:
        texto = t[1] if isinstance(t, list) else t
        if len(normalizar_texto(texto).split()) >= 3:          # frases de control cortas («no», «ya lo hice») no aplican
            frases.setdefault(texto, f"batería {c['id']}")
for c in conversaciones:
    for t in c["turnos"]:
        if len(normalizar_texto(t["usuario"]).split()) >= 3:
            frases.setdefault(t["usuario"], f"conversación {c['id']}")

modelo = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
emb = lambda xs: modelo.encode([normalizar_texto(x) for x in xs], convert_to_tensor=True, normalize_embeddings=True, batch_size=128)
lista = list(frases)
sims = util.cos_sim(emb(lista), emb(referencia))
maximos, indices = sims.max(dim=1)
fugas = [(frases[f], f, referencia[int(j)], float(s)) for f, s, j in zip(lista, maximos, indices) if float(s) >= SIM_MAX]
print(f"Frases revisadas: {len(lista)} | similitud máxima: {float(maximos.max()):.3f} | con similitud >= {SIM_MAX}: {len(fugas)}")
for origen, f, r, s in sorted(fugas, key=lambda x: -x[3]):
    print(f"  {s:.3f}  [{origen}] «{f}»  ~  «{r}»")
