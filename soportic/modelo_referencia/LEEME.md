# modelo_referencia (BETO de referencia del filtro de aumento)

BETO sin aumento de datos, learning rate 2e-5, entrenado solo con la partición de entrenamiento. Se usa como filtro de etiqueta del aumento de datos (sección 4 del notebook). No forma parte del bot.

Los pesos (`model.safetensors`, ~440 MB) y el tokenizador no se suben al repositorio.

## Cómo se genera
Si la carpeta no existe, la celda «BETO de referencia» de `SoporTIC.py` lo entrena y lo guarda aquí (`beto_ref.save_pretrained(MODELO_REFERENCIA)`). Basta con ejecutar el notebook completo. El modelo base se descarga de Hugging Face: https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased
