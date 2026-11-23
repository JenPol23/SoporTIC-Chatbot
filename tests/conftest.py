"""Construye, en CPU, el espacio de nombres del bot a partir de las celdas del notebook (fuente jupytext).

Ejecuta las celdas de configuración y datos, las definiciones de funciones del resto y las celdas «# @bot», salvo la
carga del modelo guardado y la interfaz. El clasificador se sustituye por uno de palabras clave determinista: estas
pruebas verifican la memoria del caso, los slots y los filtros, no la exactitud del modelo.
"""
import os, re, sys, io, ast, contextlib
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUENTE = os.environ.get("BOT_FUENTE", os.path.join(RAIZ, "SoporTIC.py"))
COMPLETAS = ["EN_COLAB = ", "===================== Importaciones", "Dataset propio, respuestas fijas y base de conocimiento",
             "Funciones de normalización para el EDA", "def normalizar_texto(texto: str)", "Tokenizador de BERT en español (BETO)"]
CONSTANTES = {"PATRON_SEGMENTOS", "MARCADOR_COMPLETAR", "CONTACTO_POR_DEFECTO", "INTENCIONES_TECNICAS"}
OMITIR_BOT = ["Carga de los componentes del bot", "Interfaz Gradio"]

# Clasificador sustituto: intenciones por palabra clave, en el orden en que aparecen en el mensaje
PALABRAS = [(r"contrase|clave", "restablecer_contrasena"), (r"wifi|wi-fi", "wifi_sede"), (r"teams", "teams"),
            (r"correo|outlook", "correo_institucional"), (r"excel|word|onedrive|office", "office365"),
            (r"python|pip\b|jupyter", "instalar_python"), (r"colab", "colab_gpu"), (r"academusoft|plataforma", "plataforma_academica"),
            (r"\bsala\b", "salas_computo"), (r"soporte|tecnico", "hablar_con_soporte"), (r"entregar|subir|quiz", "moodle_entregas"),
            (r"moodle|aula virtual", "moodle_acceso"), (r"^hola\b|^buenas\b", "saludo"), (r"gracias", "agradecimiento"),
            (r"\bchao\b|adios", "despedida"), (r"pelicula|futbol|receta", "fuera_de_dominio")]


def clasificador_falso(texto):
    t = texto.lower()
    hallados = sorted((m.start(), intent) for patron, intent in PALABRAS for m in [re.search(patron, t)] if m)
    intenciones = list(dict.fromkeys(i for _, i in hallados))
    if not intenciones:
        return [], {}, False, ["moodle_acceso", "correo_institucional"]
    return intenciones, {i: 0.95 for i in intenciones}, True, (intenciones + ["moodle_acceso"])[:2]


def _definiciones(src, ns):
    arbol = ast.parse(src)
    for nodo in arbol.body:
        if isinstance(nodo, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)) or (
                isinstance(nodo, ast.Assign) and any(getattr(t, "id", None) in CONSTANTES for t in nodo.targets)):
            try:
                exec(compile(ast.Module(body=[nodo], type_ignores=[]), "definicion", "exec"), ns)
            except Exception:
                pass


@pytest.fixture(scope="session")
def ns():
    os.chdir(RAIZ)
    os.environ["SOPORTIC_USAR_GPT"], os.environ["SOPORTIC_GRADIO"] = "0", "0"
    import torch
    torch.cuda.is_available = lambda: False           # todo en CPU
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: plt.close("all")
    plt.savefig = lambda *a, **k: None
    matplotlib.figure.Figure.savefig = lambda *a, **k: None
    import jupytext
    nb = jupytext.read(FUENTE)
    espacio = {"__name__": "__main__", "display": lambda *a, **k: None}
    hay_bot = False
    for celda in nb.cells:
        if celda.cell_type != "code":
            continue
        src = celda.source
        es_bot = src.lstrip().startswith("# @bot")
        if es_bot and any(m in src for m in OMITIR_BOT):
            if "Carga de los componentes del bot" in src:
                espacio.update(METODO_FINAL="BETO multietiqueta nuevo", UMBRALES_BOT=None,
                               clasificar_mensaje=clasificador_falso)
            continue
        if es_bot or any(m in src for m in COMPLETAS):
            hay_bot |= es_bot
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, "celda", "exec"), espacio)
        else:
            try:
                _definiciones(src, espacio)
            except SyntaxError:
                pass
    if not hay_bot:
        pytest.skip(f"{FUENTE} no tiene celdas «# @bot»")
    espacio["clasificar_mensaje"] = clasificador_falso
    return espacio


@pytest.fixture
def bot(ns):
    return ns["SoporTIC"]()
