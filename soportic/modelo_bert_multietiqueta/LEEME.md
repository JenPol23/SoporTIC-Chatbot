# modelo_bert_multietiqueta (v2, multietiqueta con frases concatenadas)

BETO multietiqueta de la Fase 3 (v2). Lo guarda la sección 11 del notebook (`model_multi.save_pretrained(MODEL_DIR_MULTI)` en `SoporTIC.py`), junto con `umbrales.json`.

Los pesos (`model.safetensors`, ~440 MB) y el tokenizador no se suben al repositorio; `umbrales.json` sí.

## Cómo se genera
Ejecutando el notebook completo (secciones 10–14). El modelo base se descarga de Hugging Face: https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased
