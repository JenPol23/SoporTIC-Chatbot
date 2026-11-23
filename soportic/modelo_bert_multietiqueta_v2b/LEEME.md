# modelo_bert_multietiqueta_v2b (modelo que usa el bot)

BETO (`dccuchile/bert-base-spanish-wwm-uncased`) ajustado como clasificador multietiqueta de 17 intenciones (sigmoide + BCEWithLogitsLoss), versión v2b. Lo cargan `iniciar_soportic.py` y la sección 20 del notebook (`SoporTIC.py`, celda «Carga de los componentes del bot»).

Los pesos (`model.safetensors`, ~440 MB) y el tokenizador no se suben al repositorio. Se sube `umbrales.json` (método final y umbrales por intención elegidos en validación).

## Cómo se genera
Ejecutando el notebook completo: las secciones 15–17 entrenan el modelo y la línea `model_v2b.save_pretrained(MODEL_DIR_V2B)` de `SoporTIC.py` lo guarda aquí.

```powershell
jupyter nbconvert --to notebook --execute SoporTIC.ipynb --output SoporTIC.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=soportic
```

El modelo base se descarga automáticamente de Hugging Face la primera vez: https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased

Sin esta carpeta, `iniciar_soportic.py` se detiene con «No existe el modelo guardado».
