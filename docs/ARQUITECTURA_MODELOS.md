# Modelos y conjuntos de datos de SoporTIC

Parámetros contados en CPU con `herramientas/contar_parametros.py` (salida completa en `docs/parametros_modelos.json`): `model.num_parameters()` para BETO y MarianMT, suma de los tensores para MiniLM, cabeceras de los archivos safetensors para Qwen (sin cargar los pesos) y fórmula exacta sobre el vocabulario reconstruido con los mismos datos de entrenamiento del notebook (309 frases de entrenamiento + 100 de MASSIVE = 409) para la BiLSTM y TF-IDF. Las arquitecturas salen del `config.json` de cada modelo. Las licencias son las declaradas en la ficha de Hugging Face de cada repositorio, consultadas el 2026-11-14.

## Modelos

| Modelo | Arquitectura (de `config.json`) | Parámetros | Función en el sistema | Entrenado o preentrenado | Licencia y URL |
|---|---|---|---|---|---|
| BETO (`dccuchile/bert-base-spanish-wwm-uncased`), v1 | Codificador BERT-base: 12 capas, oculta 768, 12 cabezas, FFN 3072, vocabulario 31 002; cabeza lineal de 17 salidas con softmax | 109 863 953 | Clasificador de una intención (línea base v1), filtro de etiqueta del aumento (BETO de referencia) y filtro de las frases de Qwen de entrenamiento (v1 congelada) | Preentrenado + *fine-tuning* con el dataset propio, el aumento y MASSIVE | La ficha no declara licencia en los metadatos; su texto indica «The license CC BY 4.0 best describes our intentions for our work». https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased |
| BETO multietiqueta (v2 y v2b) | Mismo codificador; 17 salidas sigmoide, `problem_type = multi_label_classification`, pérdida BCEWithLogits | 109 863 953 | Clasificador multiintención del bot (con el método final elegido por validación cruzada) | Preentrenado + *fine-tuning* multietiqueta (dataset, aumento, MASSIVE y frases sintéticas de dos intenciones) | Igual que BETO |
| MiniLM multilingüe (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) | 12 capas, oculta 384, 12 cabezas, FFN 1536, vocabulario 250 037; embedding de frase de 384 dimensiones | 117 653 760 | Respaldo semántico del híbrido v1, filtros anti-fuga (similitud ≥ 0,90), conservación de etiqueta del aumento y de las frases de Qwen, índice de fragmentos de las guías | Preentrenado (sin ajuste) | Apache-2.0. https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Qwen2.5-3B-Instruct | Decodificador: 36 capas, oculta 2048, 16 cabezas de consulta y 2 de clave/valor (GQA), FFN 11 008, vocabulario 151 936, contexto 32 768 | 3 085 938 688 | Redacción de frases de dos intenciones (datos de la Fase 3b) y modo IA del bot (redacta los pasos de la guía; un verificador controla la salida) | Preentrenado (sin ajuste); bf16 en la RTX 5050, fp32 en la T4 | `license: other`, `license_name: qwen-research` (Qwen Research License). https://huggingface.co/Qwen/Qwen2.5-3B-Instruct |
| Qwen2.5-1.5B-Instruct (respaldo) | 28 capas, oculta 1536, 12 cabezas de consulta y 2 de clave/valor, FFN 8960, vocabulario 151 936 | 1 543 714 304 | Respaldo de Qwen 3B si no hay memoria | Preentrenado (sin ajuste) | Apache-2.0. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| MarianMT es→en (`Helsinki-NLP/opus-mt-es-en`) | Transformer codificador-decodificador: 6 + 6 capas, d_model 512, 8 cabezas, FFN 2048, vocabulario 65 001 | 77 943 296 | Retrotraducción del aumento de datos (ida) | Preentrenado (sin ajuste) | Apache-2.0. https://huggingface.co/Helsinki-NLP/opus-mt-es-en |
| MarianMT en→es (`Helsinki-NLP/opus-mt-en-es`) | Igual que el anterior | 77 943 296 | Retrotraducción del aumento de datos (vuelta) | Preentrenado (sin ajuste) | Apache-2.0. https://huggingface.co/Helsinki-NLP/opus-mt-en-es |
| BiLSTM desde cero (PyTorch) | Embedding 880 × 128 (vocabulario propio), LSTM bidireccional de 128 unidades por dirección, promedio enmascarado, *dropout* 0,3, lineal 256 → 17 | 381 201 | Línea base de red neuronal sin preentrenamiento (v1) | Entrenada desde cero | Código propio |
| TF-IDF + regresión logística (scikit-learn) | 2733 características (unigramas y bigramas, `sublinear_tf`), regresión logística multinomial (C = 10) | 46 478 coeficientes | Línea base clásica (v1) | Entrenada desde cero | Código propio (scikit-learn, BSD-3-Clause) |

## Notas de licencia
- **Qwen2.5-3B-Instruct** se distribuye con la licencia **Qwen Research License** (`license_name: qwen-research`, https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE): permite el uso para investigación y fines no comerciales. SoporTIC es un proyecto académico sin uso comercial, así que el uso es compatible. Un despliegue comercial necesitaría otra licencia o el respaldo Qwen2.5-1.5B-Instruct (Apache-2.0), que el sistema ya soporta.
- **BETO**: la ficha de Hugging Face no declara licencia en sus metadatos, pero su texto dice «The license CC BY 4.0 best describes our intentions for our work» (https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased). Se trata como **CC BY 4.0**, con atribución a Cañete et al. (2020).
- **Amazon MASSIVE**: el conjunto original de Amazon declara **CC BY 4.0** (https://huggingface.co/datasets/AmazonScience/massive). El notebook descarga el espejo de MTEB (`mteb/amazon_massive_intent`), cuya ficha declara **Apache-2.0** (https://huggingface.co/datasets/mteb/amazon_massive_intent). Como el contenido es el del original, se cita y se atribuye según **CC BY 4.0** (FitzGerald et al., 2022), y se menciona el espejo como fuente de descarga.

## Recursos de la interfaz (locales, sin internet)

| Recurso | Archivos | Licencia y URL |
|---|---|---|
| Atkinson Hyperlegible Next (Braille Institute, Applied Design Works y colaboradores), fuente variable 200–800, normal e itálica, subconjuntos latin y latin-ext | `ui/fuentes/AtkinsonHyperlegibleNext-*.woff2` | **SIL Open Font License 1.1** (`ui/fuentes/OFL_atkinsonhyperlegiblenext.txt`). https://github.com/googlefonts/atkinson-hyperlegible-next · https://github.com/google/fonts/tree/main/ofl/atkinsonhyperlegiblenext |
| Atkinson Hyperlegible Mono, fuente variable 200–800, latin y latin-ext | `ui/fuentes/AtkinsonHyperlegibleMono-*.woff2` | **SIL Open Font License 1.1** (`ui/fuentes/OFL_atkinsonhyperlegiblemono.txt`). https://github.com/googlefonts/atkinson-hyperlegible-next-mono · https://github.com/google/fonts/tree/main/ofl/atkinsonhyperlegiblemono |
| Logo de la Universidad de Cundinamarca | `ui/logo_udec.png` | Marca institucional; se usa solo para identificar el proyecto académico, con el aviso de que no es un canal oficial |
| Avatares del bot y del estudiante | `soportic/ui/avatar_*.svg` (generados por el notebook) | Propios |

La OFL permite usar, incrustar y redistribuir las fuentes (también dentro del zip de Colab) siempre que se conserve el aviso de copyright y la licencia, y prohíbe venderlas por separado. Las fuentes se incrustan en el CSS de la interfaz como `data:` URI a partir de `ui/fuentes/fuentes.css`, con pila de respaldo `system-ui, Segoe UI, Arial, sans-serif`; la interfaz no pide ningún recurso a internet.

## Conjuntos de datos

| Conjunto | Contenido | Uso | Licencia y URL |
|---|---|---|---|
| Dataset propio (`dataset_soporte.csv`) | 442 frases, 17 intenciones × 26, redactadas con un asistente de IA y aprobadas por los autores | Entrenamiento, validación y prueba (70/15/15, semilla 42) | Propio de los autores |
| Generalización (`frases_generalizacion.csv`) | 68 frases (4 por intención) | Solo evaluación | Propio de los autores |
| Prueba externa (`frases_prueba_externa.txt`) | 120 líneas: 85 de una intención, 30 de varias y 5 de control con conectores; redactadas por otro asistente de IA y entregadas por los autores | Solo evaluación (una sola vez, con el método final) | Propio de los autores |
| Base de conocimiento (`base_conocimiento.json`) | 12 guías con pasos y referencias oficiales (UdeC, Microsoft, Google, Python, Moodle) | Respuestas del bot y contexto del modo IA | Propia; cada guía cita sus fuentes |
| Amazon MASSIVE, español (`mteb/amazon_massive_intent`, configuración `es`) | 180 frases de 12 escenarios ajenos al soporte (100 de entrenamiento, 30 de validación, 50 de prueba aparte) | Ejemplos de `fuera_de_dominio` | Original de Amazon: `cc-by-4.0` (https://huggingface.co/datasets/AmazonScience/massive). El espejo de MTEB que descarga el notebook declara `apache-2.0` en su ficha local (https://huggingface.co/datasets/mteb/amazon_massive_intent) |
| Frases sintéticas (v2 y v2b) | Concatenadas, negativos con conector y frases de Qwen, generadas dentro de cada partición | Entrenamiento, validación y prueba multietiqueta | Derivadas del dataset propio |
