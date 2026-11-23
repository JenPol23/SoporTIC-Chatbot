# Figuras para el informe IEEE (una columna, 3,5 pulgadas de ancho, 300 dpi, fondo blanco):
#   figuras/fig_arquitectura_soportic.png  flujo de un mensaje en el sistema
#   figuras/fig_datos_soportic.png         frases por intención y por origen, leídas de los archivos reales
# Solo usa la CPU. Uso: python herramientas/figuras_informe.py
import os, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RAIZ)
SALIDA = "figuras"
os.makedirs(SALIDA, exist_ok=True)
DATOS = os.path.join("soportic", "datos")
TINTA, TINTA_2, BORDE = "#0b0b0b", "#52514e", "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # paleta categórica validada, orden fijo
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": BORDE, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA_2, "ytick.color": TINTA_2, "text.color": TINTA, "savefig.facecolor": "white"})


# ---------------------------------------------------------------- arquitectura
def caja(ax, x, y, w, h, titulo, detalle="", color="#eef4fc", borde="#2a78d6", forma="round"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"{forma},pad=0.012,rounding_size=0.02",
                                fc=color, ec=borde, lw=0.8))
    if detalle:
        ax.text(x, y + h * 0.18, titulo, ha="center", va="center", fontsize=6.6, fontweight="bold")
        ax.text(x, y - h * 0.2, detalle, ha="center", va="center", fontsize=5.4, color=TINTA_2, linespacing=1.15)
    else:
        ax.text(x, y, titulo, ha="center", va="center", fontsize=6.6, fontweight="bold")


def flecha(ax, x0, y0, x1, y1, texto="", lado="right"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=TINTA_2, shrinkA=0, shrinkB=0))
    if texto:
        ax.text((x0 + x1) / 2 + (0.015 if lado == "right" else -0.015), (y0 + y1) / 2, texto, fontsize=5.4, color=TINTA_2,
                ha="left" if lado == "right" else "right", va="center")


def figura_arquitectura():
    fig, ax = plt.subplots(figsize=(3.5, 6.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.03); ax.axis("off")
    X, W, H = 0.56, 0.8, 0.075                     # columna principal (0,16–0,96); a la izquierda, los seguimientos
    y = {"msg": 0.95, "fil": 0.84, "mem": 0.72, "cls": 0.60, "dec": 0.485, "guia": 0.365, "modos": 0.225, "ui": 0.075}
    ax.text(0.5, 1.025, "Arquitectura de SoporTIC: flujo de un mensaje", ha="center", va="top", fontsize=7.5, fontweight="bold")
    caja(ax, X, y["msg"], W, 0.05, "Mensaje del estudiante (sesión por pestaña)", color="#f4f4f2", borde=BORDE)
    caja(ax, X, y["fil"], W, H, "Filtros de entrada",
         "contraseñas y datos personales (se ocultan)\ninyección · insultos · otro idioma · integridad · sin sentido")
    caja(ax, X, y["mem"], W, H, "Contexto y memoria del caso",
         "casos abiertos · plataforma, dispositivo, navegador, error\npaso actual · pasos dados · intentos fallidos")
    caja(ax, X, y["cls"], W, H, "Clasificación multiintención (BETO)",
         "BETO multietiqueta + segmentación por conectores\numbrales por intención elegidos en validación")
    ax.add_patch(plt.Polygon([[0.47, y["dec"] + 0.045], [0.73, y["dec"]], [0.47, y["dec"] - 0.045], [0.21, y["dec"]]],
                             fc="#fdf3e0", ec="#eda100", lw=0.8))
    ax.text(0.47, y["dec"], "¿Alguna intención\nsupera su umbral?", ha="center", va="center", fontsize=6, fontweight="bold")
    caja(ax, 0.88, y["dec"], 0.18, 0.08, "Aclaración", "dos opciones\n→ interfaz", color="#fdf3e0", borde="#eda100")
    caja(ax, X, y["guia"], W, H, "Guía paso a paso",
         "2 pasos y «¿Funcionó?» · un caso por intención\n2 fallos o pasos agotados → TICS a un CLIC + resumen")
    caja(ax, 0.355, y["modos"], 0.395, 0.1, "Modo verificado", "pasos de la guía\n+ fuente oficial", color="#e8f6f0", borde="#1baf7a")
    caja(ax, 0.765, y["modos"], 0.395, 0.1, "Modo IA", "Qwen2.5-3B + RAG de la guía\n+ verificador (si rechaza:\ntexto verificado)",
         color="#fdeee8", borde="#eb6834")
    caja(ax, X, y["ui"], W, H, "Interfaz Gradio",
         "streaming · votos · nueva conversación\ndetalles: intención, confianza y estado del caso", color="#f4f4f2", borde=BORDE)
    flecha(ax, X, y["msg"] - 0.025, X, y["fil"] + H / 2)
    flecha(ax, X, y["fil"] - H / 2, X, y["mem"] + H / 2, "mensaje válido")
    flecha(ax, X, y["mem"] - H / 2, X, y["cls"] + H / 2, "problema nuevo")
    flecha(ax, 0.47, y["cls"] - H / 2, 0.47, y["dec"] + 0.045)
    flecha(ax, 0.73, y["dec"], 0.79, y["dec"])
    ax.text(0.755, y["dec"] + 0.012, "no", fontsize=5.4, color=TINTA_2, ha="center", va="bottom")
    flecha(ax, 0.47, y["dec"] - 0.045, 0.47, y["guia"] + H / 2, "sí")
    # Seguimientos: van de la memoria a la guía sin pasar por el clasificador
    ax.plot([0.16, 0.09, 0.09], [y["mem"], y["mem"], y["guia"]], color=TINTA_2, lw=0.8)
    flecha(ax, 0.09, y["guia"], 0.16 - 0.004, y["guia"])
    ax.text(0.055, (y["mem"] + y["guia"]) / 2, "seguimiento: «no», «ya lo hice»,\n«ya funcionó», «y en el celular?»",
            rotation=90, fontsize=5.2, color=TINTA_2, ha="center", va="center")
    flecha(ax, 0.355, y["guia"] - H / 2, 0.355, y["modos"] + 0.05)
    flecha(ax, 0.765, y["guia"] - H / 2, 0.765, y["modos"] + 0.05)
    flecha(ax, 0.355, y["modos"] - 0.05, 0.46, y["ui"] + H / 2)
    flecha(ax, 0.765, y["modos"] - 0.05, 0.66, y["ui"] + H / 2)
    ruta = os.path.join(SALIDA, "fig_arquitectura_soportic.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------- datos
def leer_externo(ruta="frases_prueba_externa.txt"):
    filas = []
    for linea in open(ruta, encoding="utf-8"):
        linea = linea.strip()
        if linea and not linea.startswith("#") and ";" in linea:
            frase, et = linea.rsplit(";", 1)
            filas.append(et.strip().split("|"))
    return filas


def figura_datos():
    ds = pd.read_csv("dataset_soporte.csv")
    gen = pd.read_csv("frases_generalizacion.csv")
    ext = leer_externo()
    massive = pd.read_csv(os.path.join(DATOS, "massive_fuera_de_dominio.csv"))
    intenciones = sorted(ds["intencion"].unique())
    tabla = pd.DataFrame(index=intenciones)
    tabla["Dataset propio"] = ds["intencion"].value_counts()
    tabla["Generalización"] = gen["intencion"].value_counts()
    tabla["Externo (una intención)"] = pd.Series([e[0] for e in ext if len(e) == 1]).value_counts()
    tabla["Externo (varias, apariciones)"] = pd.Series([i for e in ext if len(e) > 1 for i in e]).value_counts()
    tabla["MASSIVE (fuera de dominio)"] = 0
    tabla.loc["fuera_de_dominio", "MASSIVE (fuera de dominio)"] = len(massive)
    tabla = tabla.fillna(0).astype(int).sort_index(ascending=False)     # orden alfabético de arriba abajo

    totales = {"Dataset propio": len(ds), "Generalización": len(gen),
               "Externo, una intención": sum(len(e) == 1 for e in ext), "Externo, varias": sum(len(e) > 1 for e in ext),
               "MASSIVE": len(massive)}
    v2 = os.path.join(DATOS, "sinteticas_dos_intenciones.csv")
    if os.path.exists(v2):
        totales["Concatenadas (v2)"] = len(pd.read_csv(v2))
    v2b = os.path.join(DATOS, "sinteticas_v2b.csv")
    if os.path.exists(v2b):
        d = pd.read_csv(v2b)
        nombres = {"concatenada_v2b": "Concatenadas (v2b)", "qwen_dos_intenciones": "Qwen, dos intenciones (v2b)",
                   "negativo_conector": "Negativos con conector (v2b)"}
        for origen, n in d["origen"].value_counts().items():
            totales[nombres.get(origen, origen)] = int(n)

    fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 6.4), gridspec_kw={"height_ratios": [1.55, 1]})
    izquierda = pd.Series(0, index=tabla.index)
    for k, col in enumerate(tabla.columns):
        a.barh(tabla.index, tabla[col], left=izquierda, color=SERIES[k], edgecolor="white", linewidth=0.6, height=0.72, label=col)
        izquierda += tabla[col]
    for i, (intent, total) in enumerate(izquierda.items()):
        a.text(total + 2, i, str(total), va="center", fontsize=5.4, color=TINTA_2)
    a.set_xlabel("Frases (externo de varias: apariciones de la intención)")
    a.set_title("(a) Frases por intención y origen", fontsize=7, loc="left", fontweight="bold")
    a.tick_params(axis="y", labelsize=5.8)
    a.legend(fontsize=5.2, frameon=False, loc="lower right")
    a.grid(axis="x", color="#e6e5e0", lw=0.5); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.set_xlim(0, izquierda.max() * 1.12)

    nombres_b = list(totales)[::-1]
    valores = [totales[n] for n in nombres_b]
    b.barh(nombres_b, valores, color=SERIES[0], height=0.65)
    for i, v in enumerate(valores):
        b.text(v + max(valores) * 0.01, i, str(v), va="center", fontsize=5.6, color=TINTA_2)
    b.set_xlim(0, max(valores) * 1.12)
    b.set_xlabel("Frases")
    b.set_title("(b) Total por origen (incluye frases sintéticas)", fontsize=7, loc="left", fontweight="bold")
    b.tick_params(axis="y", labelsize=6)
    b.grid(axis="x", color="#e6e5e0", lw=0.5); b.set_axisbelow(True)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    fig.tight_layout()
    ruta = os.path.join(SALIDA, "fig_datos_soportic.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    tabla.to_csv(os.path.join(SALIDA, "fig_datos_soportic_tabla.csv"))
    return ruta, totales


if __name__ == "__main__":
    print(figura_arquitectura())
    ruta, totales = figura_datos()
    print(ruta, totales)
