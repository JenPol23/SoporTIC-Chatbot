# Crea soportic_datos.zip para ejecutar SoporTIC.ipynb en Google Colab sin regenerar nada costoso:
# datos fuente de los autores, archivos de prueba (conversaciones, batería, tareas), la caché del aumento (MarianMT) y la
# caché de las generaciones de Qwen (Fase 3b). No incluye modelos: el notebook los entrena en Colab.
# Uso: python herramientas/crear_zip_colab.py [ruta_de_salida]
import os, sys, zipfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RAIZ)
SALIDA = sys.argv[1] if len(sys.argv) > 1 else "soportic_datos.zip"
ARCHIVOS = [
    # Datos fuente (no se regeneran)
    "dataset_soporte.csv", "frases_generalizacion.csv", "frases_prueba_externa.txt", "base_conocimiento.json",
    "respuestas_fijas.json", "dataset_soporte_multietiqueta.csv", "frases_generalizacion_multietiqueta.csv",
    "guia_anotacion_multietiqueta.md",
    # Pruebas escritas antes de ejecutar
    "conversaciones_memoria.json", "bateria_soporte.json", "tareas_evaluacion.csv",
    # v4: base anterior (secciones v1 y v2), conversaciones y batería v4, y las cifras v3 reportadas con las que se compara
    "base_conocimiento_v3.json", "conversaciones_memoria_v4.json", "bateria_soporte_v4.json",
    "soportic/resultados_soporte_v3.json", "soportic/datos/bateria_resultados_v3.csv",
    "herramientas/analizar_evaluacion_humana.py", "ui/logo_udec.png",
    # Fuentes locales de la interfaz (SIL OFL 1.1): la interfaz funciona sin internet
    "ui/fuentes/fuentes.css", "ui/fuentes/OFL_atkinsonhyperlegiblenext.txt", "ui/fuentes/OFL_atkinsonhyperlegiblemono.txt",
    *sorted(os.path.join("ui", "fuentes", f).replace(os.sep, "/") for f in os.listdir(os.path.join("ui", "fuentes")) if f.endswith(".woff2")),
    # Cachés (reproducibilidad y tiempo)
    "soportic/datos/aumento_candidatos.csv", "soportic/datos/aumento_train.csv",
    "soportic/datos/aumento_reporte.csv", "soportic/datos/qwen_generaciones_cache.csv",
]
faltan = [a for a in ARCHIVOS if not os.path.exists(a)]
if faltan:
    sys.exit(f"Faltan archivos: {faltan}")
with zipfile.ZipFile(SALIDA, "w", zipfile.ZIP_DEFLATED) as z:
    for a in ARCHIVOS:
        z.write(a, a)
print(f"{SALIDA}: {len(ARCHIVOS)} archivos, {os.path.getsize(SALIDA) / 1e6:.1f} MB")
