# v1_congelada/modelo_bert_intenciones (línea base v1 congelada)

Copia congelada del BETO de una intención de la Fase 2 (v1). Es la referencia de todas las comparaciones: la prueba externa de la v1 y el filtro de entrenamiento de las frases de Qwen (Fase 3b) usan este modelo si existe.

Los pesos (`model.safetensors`, ~440 MB) y el tokenizador no se suben al repositorio. Los resultados congelados sí se suben (`soportic/v1_congelada/resultados_soporte_v1.json`).

## Cómo se genera
El notebook no escribe en esta carpeta. Se creó copiando `soportic/modelo_bert_intenciones/` al cerrar la Fase 2. En otro equipo:

1. Ejecutar el notebook completo una vez (genera `soportic/modelo_bert_intenciones/`).
2. Copiar esa carpeta aquí:
   ```powershell
   Copy-Item soportic\modelo_bert_intenciones\* soportic\v1_congelada\modelo_bert_intenciones\ -Exclude LEEME.md
   ```

Si la carpeta no existe, el notebook usa la v1 de la ejecución actual (sección 15: «si no está, la v1 de esta ejecución, que la reproduce»). Que los pesos reentrenados den las mismas predicciones que los congelados: POR VERIFICAR (depende de la GPU y de las versiones).
