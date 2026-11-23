# Guía de cambios de SoporTIC

Recetas para instalar, modificar y volver a evaluar el proyecto. Las referencias `archivo:línea` corresponden a la versión de este repositorio (2026-11-23); si se edita el archivo, las líneas se desplazan: busque el texto citado.

Todos los comandos son de PowerShell y se ejecutan en la raíz del proyecto, con el entorno activo (`.\.venv\Scripts\Activate.ps1`).

> **Antes de cambiar datos o modelos.** Las cifras de `README.md` y `RESULTADOS_FINALES_SOPORTE.md` corresponden a los datos y modelos actuales. Cualquier cambio en el dataset, la base de conocimiento o el modelo obliga a volver a ejecutar el notebook y a reportar las cifras nuevas como otra versión. Los umbrales e hiperparámetros se eligen solo en validación; no los ajuste mirando la prueba, la generalización, el externo ni la batería.

**Fuente de verdad del notebook:** `SoporTIC.py` (formato percent). Después de editarlo, sincronice el `.ipynb`:
```powershell
jupytext --sync SoporTIC.py
```
No edite el JSON de `SoporTIC.ipynb` a mano.

---

## 1. Instalar en otro PC

1. **Python 3.11** y **PyTorch con CUDA** en el Python global. Compruebe:
   ```powershell
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```
   Si no está instalado, use el comando de https://pytorch.org/get-started/locally/ para su versión de CUDA (las GPU RTX 50xx necesitan CUDA 12.8 o superior; el proyecto se probó con `2.11.0+cu128`), por ejemplo:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```
2. **Entorno virtual y dependencias** (`requirements.txt` no incluye torch):
   ```powershell
   git clone <URL_DEL_REPOSITORIO> SoporTIC
   cd SoporTIC
   python -m venv .venv --system-site-packages
   .\.venv\Scripts\Activate.ps1
   python -c "import sys; print(sys.prefix)"      # debe terminar en \.venv
   pip install -r requirements.txt
   python -m ipykernel install --user --name soportic --display-name "SoporTIC"
   ```
3. **Modelos.** El repositorio no trae pesos. Hay dos caminos:
   - **Entrenar** (45–65 min en una GPU de 8 GB; descarga BETO, MiniLM, Qwen y MASSIVE la primera vez):
     ```powershell
     $env:SOPORTIC_USAR_GPT = "1"; $env:SOPORTIC_GRADIO = "0"
     jupyter nbconvert --to notebook --execute SoporTIC.ipynb --output SoporTIC.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=soportic
     ```
   - **Copiar** desde el equipo original la carpeta `soportic\modelo_bert_multietiqueta_v2b\` (≈ 440 MB). Es la única que necesita el bot.
4. **Primer arranque con internet** (descarga a la caché el tokenizador de BETO, MiniLM y Qwen, que el lanzador busca sin conexión):
   ```powershell
   $env:SOPORTIC_EN_LINEA = "1"
   python iniciar_soportic.py
   ```
   Los siguientes arranques: doble clic en `iniciar_soportic.bat` (funciona sin internet).
5. **Comprobación sin abrir la interfaz:**
   ```powershell
   $env:SOPORTIC_SOLO_COMPROBAR = "1"; python iniciar_soportic.py
   ```
   Debe imprimir tres líneas `OK` (`iniciar_soportic.py:96-110`).

Si el `.venv` se movió de carpeta o de equipo, bórrelo y créelo de nuevo: los entornos de Python guardan rutas absolutas.

---

## 2. Cambiar o ampliar la base de conocimiento

**Archivo:** `base_conocimiento.json`. El formato está documentado en el propio archivo (`base_conocimiento.json:3`, clave `_formato`):

```
guias.<intencion> -> {titulo, resumen, palabras_clave, diagnostico: {pregunta, opciones: [2 a 4 ramas]},
                      ramas: {<rama>: {etiqueta, palabras_clave, soluciones: [{titulo, subpasos: [1 a 4], referencias: [{titulo, url}]}]}},
                      si_no_funciona, referencias_de_la_guia, ...}
```

- **Contacto de escalamiento** (nombre, canales, horario, texto): `base_conocimiento.json:7` (`contacto_por_defecto`). El notebook lo lee en `SoporTIC.py:1708`.
- **Ejemplo de guía completa:** `teams` en `base_conocimiento.json:1034`; `wifi_sede` en `base_conocimiento.json:830`.
- **Carga en el notebook:** `SoporTIC.py:227` (`BASE_CONOCIMIENTO`, `GUIAS`). `base_conocimiento_v3.json` (`SoporTIC.py:232`) es la versión anterior y solo la usan las secciones v1–v2; no la modifique.

**Agregar una solución a una rama existente:** añada un objeto a `guias.<intencion>.ramas.<rama>.soluciones` con `titulo`, `subpasos` (1 a 4) y `referencias`.

**Agregar una rama (subproblema):** cree `guias.<intencion>.ramas.<nueva_rama>` con `etiqueta` (cómo lo diría un estudiante), `palabras_clave` y `soluciones`; si debe ofrecerse como botón, añádala a `diagnostico.opciones` (máximo 4). La rama se elige por palabras clave o, si ninguna coincide, por similitud MiniLM con la `etiqueta` (umbral 0,80, `SoporTIC.py:3405`).

**Datos institucionales que no conoce:** escriba `[COMPLETAR: descripción]`. El bot omite ese subpaso y remite a TICS a un CLIC (`base_conocimiento.json`, clave `_regla_completar`). No invente horarios, correos, redes ni URL.

**Aplicar el cambio:** el bot lee el JSON al arrancar; no hace falta reentrenar. Reinicie `iniciar_soportic.bat` y corra las pruebas (sección 6). Para reportar cifras con la base nueva, vuelva a ejecutar el notebook completo (la batería, sección 24, y la memoria, sección 22, dependen de las guías).

Para regenerar el zip de Colab con la base nueva: `python herramientas/crear_zip_colab.py`.

---

## 3. Agregar frases o etiquetas al dataset y reentrenar

**Archivos:** `dataset_soporte.csv` (columnas `texto,intencion,origen`) y `frases_generalizacion.csv` (mismo formato, 4 por intención). Validaciones del notebook:
- Sin duplicados exactos: `SoporTIC.py:239-240`.
- Cada intención técnica debe tener guía y cada social, respuesta fija: `SoporTIC.py:235-236`.
- Partición estratificada 70/15/15 con semilla 42: `SoporTIC.py:481-482`.

### 3.1 Agregar frases a intenciones existentes
1. Añada filas a `dataset_soporte.csv` (sin duplicados ni frases casi iguales a las de generalización o del externo).
2. Revise conteos, duplicados y fuga:
   ```powershell
   python herramientas/verificar_datos.py
   ```
3. Regenere la versión multietiqueta (conserva la misma partición):
   ```powershell
   python herramientas/anotar_multietiqueta.py
   ```
4. Invalide las cachés que dependen del dataset y reentrene:
   ```powershell
   Remove-Item -Recurse soportic\modelo_referencia          # el notebook lo omite si existe (SoporTIC.py:644)
   $env:SOPORTIC_REGENERAR_AUMENTO = "1"                    # SoporTIC.py:678; regenera el aumento con MarianMT
   $env:SOPORTIC_USAR_GPT = "1"; $env:SOPORTIC_GRADIO = "0"
   jupyter nbconvert --to notebook --execute SoporTIC.ipynb --output SoporTIC.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=soportic
   ```
   Las frases de Qwen de la v2b se leen de `soportic/datos/qwen_generaciones_cache.csv` por par de frases (`SoporTIC.py:2543`); los pares nuevos se generan con Qwen en esa ejecución.
5. El modelo del bot queda en `soportic/modelo_bert_multietiqueta_v2b/` (`SoporTIC.py:2910-2912`). Reinicie el lanzador.

**Comparaciones con la v1 congelada.** Si existe `soportic/v1_congelada/`, la sección 9 escribe `resultados_soporte_v1_reejecucion.json` y avisa qué cifras difieren (`SoporTIC.py:1835-1849`). Con un dataset distinto, las diferencias son esperables: la referencia congelada corresponde al dataset original.

### 3.2 Agregar una etiqueta (intención) nueva
Todo lo anterior, más:
1. **Guía** en `base_conocimiento.json` (si es técnica) o **respuesta fija** en `respuestas_fijas.json` (si es social). Sin ella falla el `assert` de `SoporTIC.py:236`.
2. **Nombres de la intención** que usa el bot (se accede con `[...]`: si falta, el bot falla con `KeyError`): `TEMA_INTENCION` (`SoporTIC.py:2318`), `NOMBRE_INTENCION` (`SoporTIC.py:3315`), `NOMBRE_CORTO` (`SoporTIC.py:3430`). Si es social, añádala también a `INTENCIONES_SOCIALES` (`SoporTIC.py:234`).
3. **Botón de tema frecuente** (opcional): `TEMAS_UI` (`SoporTIC.py:4499`) y su ícono en `ICONOS_SVG` (`SoporTIC.py:4341`).
4. **Frases de generalización** (4) en `frases_generalizacion.csv`.
5. **Mover la v1 congelada:** el modelo de `soportic/v1_congelada/` tiene 17 salidas y la sección 15 lo usa para filtrar las frases de Qwen (`SoporTIC.py:2610-2618`). Con otro número de etiquetas, muévalo fuera del proyecto antes de ejecutar (el notebook usará la v1 de esa ejecución). Tenga en cuenta que sin `v1_congelada/resultados_soporte_v1.json` la sección 9 sobrescribe `soportic/resultados_soporte_v1.json`: haga una copia antes.
6. **Pruebas:** añada la palabra clave de la intención al clasificador sustituto de `tests/conftest.py:18-23` y casos en `tests/test_memoria.py`.
7. Los conjuntos de evaluación (`bateria_soporte_v4.json`, `conversaciones_memoria_v4.json`, `frases_prueba_externa.txt`) no cubren la intención nueva. No los modifique después de ejecutarlos; si necesita evaluarla, escriba un conjunto nuevo **antes** de ejecutar.

---

## 4. Cambiar el modelo generador (modo IA)

**Dónde:** `SoporTIC.py:3970`
```python
MODELO_GPT, MODELO_GPT_RESPALDO = "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
```

- **Usar el 1.5B sin editar código:** `iniciar_soportic.bat 1.5B` (`iniciar_soportic.bat:6`) o `$env:SOPORTIC_GENERADOR = "1.5B"` (`SoporTIC.py:3972-3973`).
- **Otro modelo:** cambie el nombre por cualquier modelo *instruct* de Hugging Face con plantilla de chat (`apply_chat_template` con mensaje `system`, `SoporTIC.py:4093`) y que admita el rol `system`. Ejemplo: `MODELO_GPT = "Qwen/Qwen2.5-7B-Instruct"`. Compruebe su licencia y que quepa en la GPU (Qwen 3B en bf16 deja ≈ 0,6 GB libres en 8 GB).
- **Precisión:** bf16 si la GPU lo soporta; si no, fp32 (`SoporTIC.py:3992`).
- **Longitud y aviso de VRAM:** `MAX_TOKENS_IA = 320` (`SoporTIC.py:3974`) y `VRAM_MINIMA_LIBRE_GB = 0.5` (`SoporTIC.py:3975`).
- **Prompt:** `PROMPT_SISTEMA_SOPORTE` (`SoporTIC.py:3977`). El verificador (`verificar_ia`, `SoporTIC.py:4056`) rechaza URL, correos y números que no estén en la guía; si rechaza, se muestra el texto fijo.
- **Generador de las frases sintéticas de la v2b** (independiente del modo IA): `QWEN_MODELOS` en `SoporTIC.py:2521`. Cambiarlo solo afecta a pares que no estén en la caché `soportic/datos/qwen_generaciones_cache.csv`.

**Aplicar:** primer arranque con `$env:SOPORTIC_EN_LINEA = "1"` para descargar el modelo nuevo; luego `iniciar_soportic.bat`. Para reportar la latencia y la categoría «modo IA con verificador» de la batería con el modelo nuevo, vuelva a ejecutar el notebook (secciones 24–25).

---

## 5. Cambiar el puerto y la interfaz

### Puerto y acceso
| Qué | Dónde |
|---|---|
| Puerto inicial (7860) y rango de búsqueda (20 puertos) | `iniciar_soportic.py:113` (`puerto_libre(inicial=7860, intentos=20)`); mensajes en `:121` y `:144-145` |
| Dirección de escucha (solo este equipo) | `iniciar_soportic.py:150` (`server_name="127.0.0.1"`). `"0.0.0.0"` la abre a la red local; evalúe el riesgo antes. |
| Enlace público de Gradio | `iniciar_soportic.py:150` (`share=False`). En el notebook, `share=EN_COLAB` (`SoporTIC.py:4938`). |
| No abrir el navegador | `$env:SOPORTIC_SIN_NAVEGADOR = "1"` (`iniciar_soportic.py:150`) |
| Comentarios del .bat | `iniciar_soportic.bat:2-4` |

El `launch` del notebook (`SoporTIC.py:4938`) no fija puerto: Gradio usa 7860 o la variable `GRADIO_SERVER_PORT`.

### Textos, colores y componentes (`SoporTIC.py`, sección 23)
| Qué | Dónde |
|---|---|
| Nombre del bot | `NOMBRE_BOT`, `SoporTIC.py:4492` |
| Nombres de los modos | `MODO_FIJO_UI`, `MODO_IA_UI`, `SoporTIC.py:4493-4495` |
| Mensaje de bienvenida | `BIENVENIDA_UI`, `SoporTIC.py:4496` |
| Botones de temas frecuentes | `TEMAS_UI`, `SoporTIC.py:4499`; íconos en `ICONOS_SVG`, `SoporTIC.py:4341` |
| Texto «¿Cómo funciona SoporTIC?» | `COMO_FUNCIONA_UI`, `SoporTIC.py:4505` |
| Paleta (verde UdeC) y tipografía | `VERDE_UDEC` y `TEMA_UI`, `SoporTIC.py:4555-4558` |
| Hoja de estilos | `CSS_UI`, `SoporTIC.py:4584` |
| Encabezado institucional y logo | `html_encabezado`, `SoporTIC.py:4760`; logo en `ui/logo_udec.png` (`SoporTIC.py:4522`) |
| Disposición de la página | `construir_interfaz`, `SoporTIC.py:4893` |
| Fuentes locales | `ui/fuentes/` (`fuentes.css`) |

Después de editar `SoporTIC.py`: `jupytext --sync SoporTIC.py`, reinicie `iniciar_soportic.bat` (lee las celdas `# @bot` directamente del `.py`, `iniciar_soportic.py:67-93`) y corra las pruebas.

---

## 6. Correr las pruebas

| Prueba | Comando | Qué verifica | Tiempo |
|---|---|---|---|
| Unitarias (CPU) | `python -m pytest tests -q` | Memoria del caso, slots y filtros, con un clasificador sustituto de palabras clave (`tests/conftest.py:17-32`). Resultado al 2026-11-23: `157 passed in 28.18s` | < 1 min |
| Arranque del bot | `$env:SOPORTIC_SOLO_COMPROBAR = "1"; python iniciar_soportic.py` | Carga de los modelos guardados y tres mensajes (una intención, dos intenciones y un seguimiento) | 1–2 min con Qwen |
| Datos | `python herramientas/verificar_datos.py` | Conteos, duplicados, casi duplicados (MiniLM) y fuga entre dataset, generalización y externo | < 1 min |
| Fuga de las pruebas | `python herramientas/verificar_fuga_pruebas.py` | Similitud de la batería y las conversaciones con el entrenamiento | < 1 min |
| Batería, memoria, latencia (sistema completo) | Ejecutar el notebook completo (sección 1 de esta guía) | Secciones 22, 24 y 25; resultados en `soportic/resultados_soporte_v4.json` y `soportic/resultados_soporte_final.json` | 45–65 min |
| Evaluación humana | `python herramientas/analizar_evaluacion_humana.py` | Promedios y acuerdo entre evaluadores de `tareas_evaluacion.csv` (lo llenan los autores) | segundos |

Las pruebas unitarias no miden la exactitud del modelo; para eso está el notebook.

---

## Pendientes conocidos
- `herramientas/medir_latencia_ia.py:6` tiene fija la ruta del equipo de desarrollo (`R = r"C:\Users\...\ChatBot_Soporte"`), por eso está excluido del repositorio. Para incluirlo, reemplazar esa línea por `R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (requiere aprobación de los autores).
- `soportic_datos.zip` no se sube: se regenera con `python herramientas/crear_zip_colab.py`.
