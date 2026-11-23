# Cuenta, en CPU, los parámetros y lee la arquitectura (config.json) de cada modelo del sistema, para
# docs/ARQUITECTURA_MODELOS.md. Qwen se cuenta desde las cabeceras de safetensors, sin cargar los pesos.
# Uso: python herramientas/contar_parametros.py
import os, re, sys, json, glob, struct, unicodedata
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RAIZ)
CACHE = os.path.expanduser("~/.cache/huggingface/hub")
salida = {}


def snapshot(repo, archivo):
    rutas = glob.glob(os.path.join(CACHE, f"models--{repo.replace('/', '--')}", "snapshots", "*", archivo))
    return rutas[0] if rutas else None


def params_safetensors(repo):
    total = 0
    for ruta in glob.glob(os.path.join(CACHE, f"models--{repo.replace('/', '--')}", "snapshots", "*", "*.safetensors")):
        with open(ruta, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            cab = json.loads(f.read(n))
        total += sum(int(np.prod(v["shape"])) for k, v in cab.items() if k != "__metadata__")
    return total


from transformers import AutoModelForSequenceClassification, AutoConfig, MarianMTModel
for nombre, ruta in [("BETO (v1, una intención)", "soportic/v1_congelada/modelo_bert_intenciones"),
                     ("BETO multietiqueta (v2)", "soportic/fase3_concatenadas/modelo_bert_multietiqueta")]:
    m = AutoModelForSequenceClassification.from_pretrained(ruta)
    c = m.config
    salida[nombre] = {"parametros": m.num_parameters(), "capas": c.num_hidden_layers, "oculta": c.hidden_size,
                      "cabezas": c.num_attention_heads, "intermedia": c.intermediate_size, "vocabulario": c.vocab_size,
                      "salidas": c.num_labels, "problem_type": c.problem_type}
    del m

from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
c = st[0].auto_model.config
salida["MiniLM multilingüe"] = {"parametros": sum(p.numel() for p in st.parameters()), "capas": c.num_hidden_layers,
                               "oculta": c.hidden_size, "cabezas": c.num_attention_heads, "intermedia": c.intermediate_size,
                               "vocabulario": c.vocab_size, "dim_embedding": st.get_sentence_embedding_dimension()}
del st

for repo in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]:
    c = json.load(open(snapshot(repo, "config.json"), encoding="utf-8"))
    salida[repo] = {"parametros": params_safetensors(repo), "capas": c["num_hidden_layers"], "oculta": c["hidden_size"],
                    "cabezas": c["num_attention_heads"], "cabezas_kv": c.get("num_key_value_heads"),
                    "intermedia": c["intermediate_size"], "vocabulario": c["vocab_size"], "contexto": c.get("max_position_embeddings")}

for repo in ["Helsinki-NLP/opus-mt-es-en", "Helsinki-NLP/opus-mt-en-es"]:
    m = MarianMTModel.from_pretrained(repo)
    c = m.config
    salida[repo] = {"parametros": m.num_parameters(), "capas_codificador": c.encoder_layers, "capas_decodificador": c.decoder_layers,
                    "oculta": c.d_model, "cabezas": c.encoder_attention_heads, "intermedia": c.encoder_ffn_dim, "vocabulario": c.vocab_size}
    del m

# BiLSTM y TF-IDF: se reconstruye el vocabulario con los mismos datos de entrenamiento del notebook (train + 100 de MASSIVE)
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True); nltk.download("stopwords", quiet=True)
quitar = lambda t: "".join(ch for ch in unicodedata.normalize("NFD", t) if unicodedata.category(ch) != "Mn")
def normalizar_eda(t):
    t = quitar(t.lower()); t = re.sub(r"[^a-z\s]", " ", t); return re.sub(r"\s+", " ", t).strip()
def tokenizar(t):
    return [x for x in nltk.word_tokenize(normalizar_eda(t), language="spanish") if len(x) > 1]
df = pd.read_csv("dataset_soporte.csv")
etiquetas = sorted(df["intencion"].unique())
df["label"] = df["intencion"].map({e: i for i, e in enumerate(etiquetas)})
train, _ = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
massive = pd.read_csv("soportic/datos/massive_fuera_de_dominio.csv")
textos = pd.concat([train["texto"], massive[massive["uso"] == "train"]["texto"]])
vocab = {"<pad>", "<unk>"} | {w for t in textos for w in tokenizar(t)}
V, E, Hd, C = len(vocab), 128, 128, len(etiquetas)
lstm = 2 * (4 * (E * Hd + Hd * Hd + 2 * Hd))          # bidireccional, sesgos de entrada y recurrentes
salida["BiLSTM desde cero"] = {"parametros": V * E + lstm + (2 * Hd) * C + C, "vocabulario": V, "embedding": E, "oculta": Hd,
                               "direcciones": 2, "salidas": C, "frases_entrenamiento": len(textos)}
vec = TfidfVectorizer(preprocessor=normalizar_eda, ngram_range=(1, 2), sublinear_tf=True).fit(textos)
F = len(vec.vocabulary_)
salida["TF-IDF + regresión logística"] = {"parametros": F * C + C, "caracteristicas": F, "ngramas": "1–2", "salidas": C,
                                          "frases_entrenamiento": len(textos)}
json.dump(salida, open("docs/parametros_modelos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
for k, v in salida.items():
    print(k, v)
