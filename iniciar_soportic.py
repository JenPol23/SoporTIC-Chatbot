"""Inicia SoporTIC con los modelos GUARDADOS, sin entrenar nada.

Ejecuta, de SoporTIC.py (la misma fuente que el notebook):
- completas, las celdas de configuración y lectura de datos (importaciones, dataset, base de conocimiento, normalización,
  tokenizador) y todas las celdas marcadas con «# @bot» (memoria del caso, filtros, modo IA e interfaz);
- del resto, solo las definiciones de funciones y unas pocas constantes (no entrena, no evalúa, no guarda figuras).
Carga el BETO multietiqueta de soportic/modelo_bert_multietiqueta_v2b (con el método y los umbrales elegidos en la
Fase 3b) y, si SOPORTIC_USAR_GPT=1, Qwen2.5-3B (1.5B si falta memoria, o siempre con SOPORTIC_GENERADOR=1.5B,
que también activa «iniciar_soportic.bat 1.5B»). Tras cargar Qwen informa la VRAM libre y avisa si queda menos de 0,5 GB.
Antes de abrir Gradio prueba tres mensajes: una intención, dos intenciones y un seguimiento.
Lanza la interfaz en http://127.0.0.1:7860 (o en el siguiente puerto libre) y abre el navegador.
"""
import os, sys, io, ast, time, socket, contextlib

INICIO = time.time()
RAIZ = os.path.dirname(os.path.abspath(__file__))
os.chdir(RAIZ)
os.environ.setdefault("SOPORTIC_USAR_GPT", "1")
os.environ["SOPORTIC_GRADIO"] = "0"                     # la interfaz se lanza aquí, no dentro de la celda
# Funcionamiento sin internet: modelos solo desde la caché local y sin telemetría de Gradio
# (SOPORTIC_EN_LINEA=1 permite descargas si falta algo en la caché)
if os.environ.get("SOPORTIC_EN_LINEA", "0") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
FUENTE = os.environ.get("SOPORTIC_FUENTE", "SoporTIC.py")
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")
plt.savefig = lambda *a, **k: None                     # no se sobrescriben las figuras del notebook
matplotlib.figure.Figure.savefig = lambda *a, **k: None
import jupytext

MODELO_GUARDADO = os.path.join("soportic", "modelo_bert_multietiqueta_v2b")

# Celdas que se ejecutan completas (configuración y datos; no entrenan)
COMPLETAS = ["EN_COLAB = ", "===================== Importaciones", "Dataset propio, respuestas fijas y base de conocimiento",
             "Funciones de normalización para el EDA", "def normalizar_texto(texto: str)", "Tokenizador de BERT en español (BETO)"]
# Constantes que se toman de celdas que, por lo demás, entrenan o evalúan
CONSTANTES = {"PATRON_SEGMENTOS", "MARCADOR_COMPLETAR", "CONTACTO_POR_DEFECTO", "INTENCIONES_TECNICAS"}


def solo_definiciones(src):
    """Conserva de una celda solo importaciones, funciones, clases y las constantes de CONSTANTES."""
    arbol = ast.parse(src)
    arbol.body = [n for n in arbol.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
                  or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in CONSTANTES for t in n.targets))]
    return ast.unparse(arbol)


def ejecutar_definiciones(src, ns):
    """Ejecuta una a una las definiciones de una celda; omite las que dependen de algo que no se cargó
    (p. ej., un valor por defecto calculado en una celda de evaluación), porque el bot no las usa."""
    for nodo in ast.parse(solo_definiciones(src)).body:
        try:
            exec(compile(ast.Module(body=[nodo], type_ignores=[]), "definicion", "exec"), ns)
        except Exception:
            pass


def ejecutar_celdas():
    nb = jupytext.read(FUENTE)
    ns = {"__name__": "__main__", "display": lambda *objs, **k: None}
    for i, celda in enumerate(nb.cells):
        if celda.cell_type != "code":
            continue
        src = celda.source
        primera = next((l.strip("# -=").strip() for l in src.splitlines() if l.strip() and "@bot" not in l), "")[:70]
        if src.lstrip().startswith("# @bot") or any(m in src for m in COMPLETAS):
            codigo, desc = src, primera
        else:
            try:
                ejecutar_definiciones(src, ns)
            except SyntaxError:          # celdas con magias de Jupyter
                pass
            continue
        t0, salida = time.time(), io.StringIO()
        try:
            with contextlib.redirect_stdout(salida):
                exec(compile(codigo, f"celda_{i}", "exec"), ns)
        except Exception:
            print(salida.getvalue())
            print(f"ERROR en la celda {i} ({desc})")
            raise
        if desc != "solo definiciones":
            print(f"  [{time.time() - INICIO:6.1f} s] celda {i:3d} lista ({time.time() - t0:5.1f} s): {desc}", flush=True)
    return ns


def comprobar(ns):
    """Tres mensajes con un bot nuevo: una intención, dos intenciones y un seguimiento sobre el caso activo."""
    bot = ns["SoporTIC"]()
    pruebas = [("no puedo entrar a moodle desde el celular", lambda t: t["accion"] == "guiar" and len(t["casos"]) == 1),
               ("olvide la contraseña del correo y ademas teams no me abre", lambda t: t["accion"] == "guiar" and len(t["casos"]) >= 3),
               ("y en el portatil?", lambda t: t["accion"] == "cambio_slot")]
    todo_ok = True
    for mensaje, criterio in pruebas:
        respuesta = ns["responder_modo"](bot, mensaje, "fijo")
        turno = bot.historial[-1]
        ok = bool(respuesta.strip()) and criterio(turno) and "[COMPLETAR" not in respuesta
        todo_ok &= ok
        casos = [c["intencion"] for c in turno["casos"]]
        print(f"  {'OK   ' if ok else 'FALLA'} «{mensaje}» → acción {turno['accion']} | casos {casos} | slots {turno['slots']}")
    return todo_ok


def puerto_libre(inicial=7860, intentos=20):
    for p in range(inicial, inicial + intentos):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise OSError("No hay puertos libres entre 7860 y 7879.")


if __name__ == "__main__":
    print("Iniciando SoporTIC (modelos guardados, sin entrenamiento)…", flush=True)
    if not os.path.exists(os.path.join(MODELO_GUARDADO, "model.safetensors")):
        print(f"No existe el modelo guardado en {MODELO_GUARDADO}: ejecute antes el notebook completo.")
        sys.exit(1)
    ns = ejecutar_celdas()
    print(f"Clasificador: {ns['METODO_FINAL']} | modo IA: {'sí, ' + str(ns['MODELO_GPT_USADO']) if ns['USAR_GPT'] else 'no'}")
    if ns["USAR_GPT"] and ns["vram_libre_gb"]() is not None:
        libre = ns["vram_libre_gb"]()
        print(f"VRAM libre tras cargar Qwen: {libre:.2f} GB", flush=True)
        if libre < ns["VRAM_MINIMA_LIBRE_GB"]:
            print(f"AVISO: quedan menos de {str(ns['VRAM_MINIMA_LIBRE_GB']).replace('.', ',')} GB de VRAM libres. Si otro programa usa la GPU "
                  "(navegador con aceleración por hardware, juegos, otro notebook), Windows pasa parte de Qwen a la memoria "
                  "compartida y el modo IA puede tardar 20–40 s por respuesta. Cierre esos programas o, si sigue lento, "
                  "reinicie con el generador de respaldo: set SOPORTIC_GENERADOR=1.5B y vuelva a ejecutar iniciar_soportic.bat.",
                  flush=True)
    print("Comprobación interna:")
    if not comprobar(ns):
        print("AVISO: alguna comprobación no dio lo esperado; se abre la interfaz igualmente.")
    puerto = puerto_libre()
    if puerto != 7860:
        print(f"AVISO: el puerto 7860 está ocupado; se usa el {puerto}.")
    if os.environ.get("SOPORTIC_SOLO_COMPROBAR") == "1":
        print("SOPORTIC_SOLO_COMPROBAR=1: no se abre la interfaz.")
        sys.exit(0)
    demo = ns["demo"]
    demo.queue().launch(server_name="127.0.0.1", server_port=puerto, inbrowser=os.environ.get("SOPORTIC_SIN_NAVEGADOR", "0") != "1", share=False, prevent_thread_lock=True,
                        css=ns["CSS_UI"], theme=ns["TEMA_UI"], head=ns["HEAD_UI"], footer_links=[],
                        allowed_paths=ns["RUTAS_PERMITIDAS_UI"])
    print(f"SoporTIC listo en http://127.0.0.1:{puerto} (arranque: {time.time() - INICIO:.0f} s). "
          "Cierre esta ventana o pulse Ctrl+C para detenerlo.", flush=True)
    try:
        demo.block_thread()
    except KeyboardInterrupt:
        demo.close()
