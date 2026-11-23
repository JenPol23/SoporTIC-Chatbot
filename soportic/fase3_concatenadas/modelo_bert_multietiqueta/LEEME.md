# fase3_concatenadas/modelo_bert_multietiqueta (v2 conservada)

Copia del BETO multietiqueta v2 tal como quedó al cerrar la Fase 3, conservada junto con `resultados_soporte_v2.json` para comparación. El notebook no escribe en esta carpeta (la reejecución de la v2 guarda en `soportic/modelo_bert_multietiqueta/`).

Los pesos (`model.safetensors`, ~440 MB) y el tokenizador no se suben al repositorio; `umbrales.json` sí.

## Cómo se genera
Ejecutar el notebook completo y copiar `soportic/modelo_bert_multietiqueta/` aquí:

```powershell
Copy-Item soportic\modelo_bert_multietiqueta\* soportic\fase3_concatenadas\modelo_bert_multietiqueta\ -Exclude LEEME.md
```
