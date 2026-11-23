# modelo_bert_intenciones (v1, una intención, ejecución actual)

BETO (`dccuchile/bert-base-spanish-wwm-uncased`) ajustado como clasificador de una intención (17 clases), línea base v1. Lo guarda la sección 9 del notebook (`model.save_pretrained(MODEL_DIR)` en `SoporTIC.py`).

Los pesos (`model.safetensors`, ~440 MB) y el tokenizador no se suben al repositorio.

## Cómo se genera
Ejecutando el notebook completo (secciones 1–9). El modelo base se descarga de Hugging Face la primera vez: https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased
