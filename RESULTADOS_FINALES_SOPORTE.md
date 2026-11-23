# Resultados finales de SoporTIC

Todas las cifras salen de las salidas reales del notebook (ejecución final del 2026-11-15, 16:44–17:28, con la lógica v4; RTX 5050 Laptop 8 GB; las cifras de clasificación de v1, v2 y v2b son idénticas a las de las ejecuciones anteriores) o de los JSON de resultados. Cada tabla indica su archivo de origen, relativo a `soportic/`. Los criterios de evaluación y las decisiones están en `DECISIONES.md` y `docs/DECISIONES_RESUMEN.md`.

## 1. Línea base de una intención (v1, congelada)
Origen: `v1_congelada/resultados_soporte_v1.json` (prueba, generalización, validación cruzada y MASSIVE) y `resultados_soporte_v1_reejecucion.json` (prueba externa).

| Modelo | Acc. prueba (67) | F1 macro prueba | Acc. generalización (68) | F1 macro gen. | Acc. externa (90) | F1 macro externa | Fuera de dominio MASSIVE (50) |
|---|---|---|---|---|---|---|---|
| TF-IDF + regresión logística | 0,5522 | 0,5642 | 0,7059 | 0,7277 | 0,7333 | 0,7591 | 0,98 |
| BiLSTM desde cero | 0,4328 | 0,4311 | 0,5588 | 0,5950 | 0,5889 | 0,5943 | 0,92 |
| Solo similitud semántica (MiniLM) | 0,6119 | 0,6015 | 0,7794 | 0,7754 | 0,8000 | 0,7943 | 0,90 |
| BETO *fine-tuned* | 0,7612 | 0,7316 | 0,8824 | 0,8827 | 0,8111 | 0,8256 | 0,94 |
| Híbrido (BETO + semántico) | 0,7910 | 0,7811 | 0,8824 | 0,8850 | 0,8111 | 0,7774 | 0,96 |

La cifra externa de BETO se calculó con el modelo congelado; sus predicciones coinciden con las del BETO reejecutado en las 90 frases. El control de fuga del externo dio 0 frases con similitud MiniLM ≥ 0,90.

**Validación cruzada de 5 particiones (accuracy, media ± DE).**

| Modelo | v1 congelada | Reejecución |
|---|---|---|
| BETO | 0,7895 ± 0,0384 | 0,7758 ± 0,0519 |
| MiniLM | 0,6448 ± 0,0466 | 0,6448 ± 0,0466 |
| TF-IDF | 0,5701 ± 0,0450 | 0,5701 ± 0,0450 |
| BiLSTM | 0,4977 ± 0,0536 | 0,4977 ± 0,0536 |

La diferencia de BETO se debe a que el filtro anti-fuga del aumento dentro de cada pliegue incluye el externo, que cambió entre ejecuciones.

**Validación (v1).** Umbrales del híbrido: 0,45 (BETO) y 0,60 (semántico); accuracy de validación 0,8750 frente a 0,8646 solo con BETO. Aclaración con U = 0,65: el bot aclara en el 11,46 % de la validación y la accuracy de lo respondido directamente es 0,9412.

## 2. Calibración de la v1
Origen: `resultados_soporte_v2.json` (`temperature_scaling_v1`).

| Conjunto | ECE antes | ECE después (T = 1,0461) |
|---|---|---|
| Validación | 0,0657 | 0,0652 |
| Prueba | 0,1442 | 0,1411 |

## 3. Multiintención v2 (concatenadas, conservada en `fase3_concatenadas/`)
Origen: `resultados_soporte_v2.json`. La reejecución (`resultados_soporte_v2_reejecucion.json`) coincide en todas las secciones.

| Conjunto | Modelo | Exact match | F1 micro | F1 macro | Hamming | Exactitud 1 intención | Exactitud 2+ intenciones |
|---|---|---|---|---|---|---|---|
| Prueba (107) | v1 top-1 | 0,4766 | 0,6850 | 0,6696 | 0,0440 | 0,7612 | 0,0000 |
| Prueba (107) | Segmentación + v1 | 0,6729 | 0,7667 | 0,7387 | 0,0385 | 0,7463 | 0,5500 |
| Prueba (107) | BETO multietiqueta | 0,6729 | 0,8000 | 0,7898 | 0,0324 | 0,7761 | 0,5000 |
| Generalización (68) | BETO multietiqueta | 0,8971 | 0,9143 | 0,9192 | 0,0104 | 0,8971 | — |

Validación cruzada del multietiqueta v2: exact match 0,6246 ± 0,0488, F1 micro 0,7666 ± 0,0337, exactitud en 2+ intenciones 0,3250 ± 0,1275.

## 4. Multiintención v2b (diseñada después de ver la prueba de la v2)
Origen: `resultados_soporte_v2b.json` y `eleccion_metodo_v2b.json`. La elección quedó fijada el 2026-11-14 a las 23:58, antes de calcular la prueba.

**Validación cruzada (5 particiones, media ± DE; en la prueba de cada pliegue).**

| Método | Exact match | F1 micro | F1 macro | Hamming | Exactitud 1 | Exactitud 2+ | Tasa FP multiintención (prueba) | Tasa FP (validación) |
|---|---|---|---|---|---|---|---|---|
| **Ensamble (completo + segmentos)** | **0,6496 ± 0,0463** | 0,8279 ± 0,0404 | 0,8193 ± 0,0313 | 0,0321 ± 0,0075 | 0,7985 ± 0,0300 | 0,5400 ± 0,0616 | 0,0634 ± 0,0204 | 0,0933 ± 0,0418 |
| BETO multietiqueta nuevo | 0,6151 ± 0,0686 | 0,8157 ± 0,0507 | 0,8130 ± 0,0415 | 0,0332 ± 0,0092 | 0,8075 ± 0,0393 | 0,4733 ± 0,0949 | 0,0589 ± 0,0234 | 0,0687 ± 0,0322 |
| Combinado (segmentos + multietiqueta) | 0,6026 ± 0,0596 | 0,8093 ± 0,0475 | 0,7852 ± 0,0327 | 0,0377 ± 0,0098 | 0,7511 ± 0,0295 | 0,4933 ± 0,0904 | 0,1290 ± 0,0154 | 0,1990 ± 0,0603 |

**Validación (partición principal; piso de umbral elegido por exact match de validación).**

| Método | Piso | Exact match | F1 micro | Hamming | Exactitud 1 | Exactitud 2+ | Tasa FP |
|---|---|---|---|---|---|---|---|
| Ensamble | sin piso | 0,7439 | 0,8669 | 0,0232 | 0,8730 | 0,6083 | 0,0635 |
| Multietiqueta nuevo | sin piso | 0,6951 | 0,8552 | 0,0261 | 0,8254 | 0,5583 | 0,1349 |
| Combinado | piso 0,30 | 0,6585 | 0,7865 | 0,0395 | 0,7857 | 0,5250 | 0,1587 |
| Segmentación + v1 | — | 0,5894 | 0,7649 | 0,0416 | 0,7698 | 0,4000 | 0,1349 |
| v1 top-1 | — | 0,4512 | 0,7059 | 0,0430 | 0,8810 | 0,0000 | 0,0000 |

**Prueba (187 = 67 originales + 80 concatenadas + 40 de Qwen) y generalización.**

| Método | Exact match total | F1 micro | Hamming | Originales (67) | Concatenadas (80) | Qwen (40) | Tasa FP (originales) | Generalización (68) |
|---|---|---|---|---|---|---|---|---|
| **Ensamble** | **0,6578** | 0,8293 | 0,0330 | 0,7761 | 0,5875 | 0,6000 | 0,0746 | 0,9118 |
| Combinado | 0,6150 | 0,7975 | 0,0412 | 0,7463 | 0,5125 | 0,6000 | 0,1194 | 0,8971 |
| Segmentación + v1 | 0,5829 | 0,7496 | 0,0475 | 0,7463 | 0,5125 | 0,4500 | 0,0448 | 0,8824 |
| Multietiqueta nuevo | 0,5455 | 0,7917 | 0,0412 | 0,7015 | 0,4500 | 0,4750 | 0,1493 | 0,8676 |
| Multietiqueta v2 | 0,4973 | 0,7372 | 0,0500 | 0,7761 | 0,3625 | 0,3000 | 0,0149 | 0,8971 |
| v1 top-1 | 0,2727 | 0,6154 | 0,0598 | 0,7612 | 0,0000 | 0,0000 | 0,0000 | 0,8824 |

**Prueba externa (una sola vez, con el ensamble).** Origen: `resultados_soporte_v2b.json` (`externo`), `datos/v2b_externo_por_linea.csv`, figura `figuras/fig20_matriz_externa_v2b.png`.

| Subconjunto | n | Exact match | F1 micro | F1 macro | Hamming | Tasa FP multiintención |
|---|---|---|---|---|---|---|
| Externo completo | 120 | 0,8000 | 0,8600 | 0,8404 | 0,0206 | — |
| Una intención | 90 | 0,8444 | 0,8370 | 0,8316 | 0,0196 | 0,0444 |
| 30 naturales de varias intenciones | 30 | 0,6667 | 0,8966 | 0,7659 | 0,0235 | — |
| 5 de control con conectores | 5 | 1,0000 | 1,0000 | 1,0000 | 0,0000 | 0,0000 |

Datos de la v2b (por partición): 600/80/80 concatenadas, 300/40/40 de Qwen y 150/30 negativos con conector. Filtros de Qwen en la partición principal: entrenamiento, 159 descartadas por el top-3 de la v1 y 6 por una tercera intención ≥ 0,30; validación y prueba, 12 y 17 por similitud con el origen < 0,50 y 0 por servicios. No se generó ninguna frase nueva con Qwen.

**v2b + normalización de tildes (fila aparte, sin reentrenar).** Decidida después de la prueba manual de la interfaz, donde «olvidé mi contraseña» se clasificaba como `agradecimiento` (DECISIONES.md n.º 40): el bot quita las tildes de las vocales antes de clasificar, con el mismo modelo y los mismos umbrales. Origen: `resultados_soporte_v2b_tildes.json`, `datos/v2b_tildes.csv`. Las cifras de la v2b de arriba no cambian.

| Conjunto | Exact match v2b | Exact match v2b + tildes | F1 micro v2b | F1 micro v2b + tildes |
|---|---|---|---|---|
| Prueba (187) | 0,6578 | 0,6631 | 0,8293 | 0,8325 |
| Generalización (68) | 0,9118 | 0,9118 | 0,9130 | 0,9130 |
| Externo completo (120) | 0,8000 | 0,8000 | — | — |
| 30 naturales de varias intenciones | 0,6667 | 0,6667 | — | — |

## 5. Memoria del caso (v3)
Origen: `resultados_soporte_v3.json`, `datos/memoria_*.csv`, figura `figuras/fig21_memoria_caso.png`. 30 conversaciones, 104 turnos.

| Métrica | Con memoria | Sin memoria |
|---|---|---|
| Intención por turno | 0,9231 | 0,3558 |
| Acción por turno | 0,9808 | 0,3558 |
| Caso activo | 0,9231 | 0,6154 |
| Slots correctos (47) | 0,9787 | 0,7872 |
| Conversaciones resueltas o bien escaladas (30) | 0,9000 | 0,0000 |
| Seguimientos bien resueltos (67) | 0,8955 | 0,0000 |
| Pasos repetidos (texto mostrado) | 2* | 4 |
| Avisos de privacidad correctos (2) | 1,0000 | 1,0000 |

\* Los 2 corresponden al resumen de escalamiento, que enumera los pasos intentados; ningún paso se dio dos veces (DECISIONES.md n.º 21). Con el resumen del rediseño de la interfaz (primera cláusula de cada paso, sin cortar palabras), la misma métrica da 0; la batería y las demás métricas de esta tabla no cambian (regresión en CPU, DECISIONES.md n.º 26). Las cifras de la tabla son las de la ejecución completa del notebook.

## 5b. Memoria del caso (v4: subproblemas, una solución por turno, cambio de tema)
Origen: `resultados_soporte_v4.json`, `datos/memoria_v4_*.csv`, figura `figuras/fig22_memoria_v4.png`. Mismos 104 turnos de las 30 conversaciones (`conversaciones_memoria_v4.json`), con 41 expectativas actualizadas de forma mecánica según las reglas v4 antes de ejecutar, más 5 conversaciones nuevas de cambio de tema (22 turnos).

| Métrica | v3 con memoria | v4 con memoria | v4 sin memoria | v4 cambio de tema (5) |
|---|---|---|---|---|
| Intención por turno | 0,9231 | 0,9231 | 0,3846 | 0,9545 |
| Acción por turno | 0,9808 | 0,9712 | 0,3558 | 0,9545 |
| Caso activo | 0,9231 | 0,9327 | 0,6635 | 0,9545 |
| Slots correctos | 0,9787 | 0,9787 | 0,8085 | 1,0000 (1) |
| Resueltas o bien escaladas | 0,9000 | 0,9000 | 0,0000 | 0,8000 |
| Seguimientos bien resueltos | 0,8955 (67) | 0,9104 (67) | 0,0000 | 0,9412 (17) |
| Pasos (v3) / soluciones (v4) repetidos | 2 | 0 | 0 | 0 |
| Cambios de tema bien resueltos | — | — | — | 1,0000 (5) |

Los errores restantes (10 turnos) vienen de los mismos 3 primeros mensajes mal clasificados que en la v3 (c05, c11, c23) y de un turno de n04, donde el último «no, sigue igual» recibe una aclaración. La comparación v3–v4 es parcial porque las expectativas cambiaron.

## 6. Batería del sistema completo (254 casos)

**v4 (ejecución final).** Origen: `resultados_soporte_final.json`, `datos/bateria_resultados.csv`; criterios de `bateria_soporte_v4.json` (8 criterios en 5 casos actualizados de forma mecánica antes de ejecutar). La v3 queda en `datos/bateria_resultados_v3.csv`.

| Categoría | Pasa | Aclara | Falla | % pasa |
|---|---|---|---|---|
| Intención (17 × 5) | 74 | 3 | 8 | 87,1 |
| Ortografía y jerga | 21 | 3 | 1 | 84,0 |
| Entrada inválida | 12 | 0 | 3 | 80,0 |
| Multiintención | 13 | 0 | 7 | 65,0 |
| Seguimientos y memoria | 19 | 0 | 1 | 95,0 |
| Escalamiento | 6 | 1 | 3 | 60,0 |
| Privacidad | 15 | 0 | 0 | 100,0 |
| Inyección de instrucciones | 10 | 0 | 0 | 100,0 |
| Otro idioma | 7 | 0 | 1 | 87,5 |
| Fuera de dominio | 9 | 1 | 0 | 90,0 |
| Trámites | 2 | 2 | 4 | 25,0 |
| `[COMPLETAR]` nunca mostrado | 12 | 0 | 0 | 100,0 |
| Sesiones paralelas | 3 | 0 | 3 | 50,0 |
| Modo IA con verificador | 10 | 0 | 0 | 100,0 |
| **Total** | **213** | **10** | **31** | **83,9** |

Fallos funcionales: 0. **Casos que cambian frente a la v3 (8):**

| Caso | Mensajes | v3 → v4 | Por qué |
|---|---|---|---|
| b039 | no recibo correos en outlook | aclara → pasa | Reglas de aclaración (DECISIONES.md n.º 50–52) |
| b041 | olvide mi clave | aclara → pasa | Reglas de aclaración |
| b093 | como instalo paiton | aclara → pasa | Reglas de aclaración |
| b147 | el correo no abre en el portatil → y en el celular tampoco? | aclara → pasa | Reglas de aclaración |
| b166 | moodle no abre → no → no | pasa → falla | El primer mensaje no indica el subproblema: el bot pregunta el diagnóstico y «no» no lo responde, así que no escala |
| b167 | la red de la u no me deja navegar → sigue igual → tampoco | pasa → falla | Igual que b166 |
| b241 | (sesiones A y B) la red inalambrica de la u me rechaza → no → no | pasa → falla | Dos «no» no agotan las soluciones de la rama; la v3 escalaba tras dos fallos |
| b244 | (A) python no instala → … → tampoco | pasa → falla | La rama de Python tiene una sola solución: «tampoco» la agota y escala; la v3 esperaba el caso activo |

Los 4 fallos nuevos vienen del diseño v4 (diagnóstico y número de soluciones por rama), no de errores funcionales; se reportan sin cambiar los criterios.

**v3 (reportada).** Origen: `datos/bateria_resultados_v3.csv` y `datos/bateria_resultados.csv`; antes de la corrección del verificador: `datos/bateria_resultados_antes.csv` y `resultados_soporte_final_antes_correccion.json`.

| Categoría | Pasa | Aclara | Falla | % pasa |
|---|---|---|---|---|
| Intención (17 × 5) | 72 | 5 | 8 | 84,7 |
| Ortografía y jerga | 20 | 4 | 1 | 80,0 |
| Entrada inválida | 12 | 0 | 3 | 80,0 |
| Multiintención | 13 | 0 | 7 | 65,0 |
| Seguimientos y memoria | 18 | 1 | 1 | 90,0 |
| Escalamiento | 8 | 1 | 1 | 80,0 |
| Privacidad | 15 | 0 | 0 | 100,0 |
| Inyección de instrucciones | 10 | 0 | 0 | 100,0 |
| Otro idioma | 7 | 0 | 1 | 87,5 |
| Fuera de dominio | 9 | 1 | 0 | 90,0 |
| Trámites | 2 | 2 | 4 | 25,0 |
| `[COMPLETAR]` nunca mostrado | 12 | 0 | 0 | 100,0 |
| Sesiones paralelas | 5 | 0 | 1 | 83,3 |
| Modo IA con verificador | 10 | 0 | 0 | 100,0 |
| **Total** | **213** | **14** | **27** | **83,9** |

Fallos funcionales: 0. Antes y después de la corrección del verificador del modo IA: los mismos estados (213/14/27). Redacciones de Qwen aceptadas por el verificador: 0/10 antes y 10/10 después.

## 7. Latencia y VRAM
**v4 (ejecución final).** Origen: `resultados_soporte_final.json`.

| Modo | n | Media (s) | p95 (s) | Máximo (s) |
|---|---|---|---|---|
| Pasos verificados (fijo) | 30 | 0,039 | 0,050 | 0,067 |
| Redactadas por IA (Qwen2.5-3B, bf16) | 30 | 1,381 | 4,785 | 5,417 |

La media del modo IA **no es comparable** con la de la v3: en la v4, 19 de los 30 mensajes de la medición reciben una pregunta de diagnóstico, una aclaración o un escalamiento, que se muestran con el texto fijo sin llamar a Qwen; solo 11 pasan por Qwen (conteo con el mismo bot en CPU). Para una respuesta redactada, el p95 y el máximo (≈ 5 s) son la referencia. Pico de VRAM: 6,98 GB. Tiempo total del notebook: 3845,2 s (64,1 min; la ejecución anterior con el mismo código, salvo la corrección n.º 61, tardó 44,0 min: el aumento está en los entrenamientos de la v1 y la v2, no en la evaluación del bot).

**v3 (reportada).**

| Modo | n | Media (s) | p95 (s) | Máximo (s) |
|---|---|---|---|---|
| Pasos verificados (fijo) | 30 | 0,021 | 0,028 | 0,030 |
| Redactadas por IA (Qwen2.5-3B, bf16) | 30 | 4,381 | 5,387 | 5,647 |

Cifras de la ejecución del notebook con la liberación de memoria de la sección 21 (DECISIONES.md n.º 24). **Antes** de esa corrección, el modo IA medía 17,235 s de media (p95 21,130 s, máximo 24,477 s; 14,832 s en la ejecución previa al renombre), copia en `resultados_soporte_final_antes_latencia.json`. La causa era la memoria residual en la GPU (modelos que el bot no usa y gradientes de `model_v2b`): Windows desbordaba parte de Qwen a la memoria compartida y la generación bajaba de unos 22 a unos 6–7 tokens/s. Es una corrección de la medición: las predicciones, la batería y las demás métricas no cambian.

Medición independiente en el proceso de la interfaz (`docs/diagnostico_latencia_ia/`, 10 frases de la demo, 2 repeticiones): Qwen 3B, media 5,89 s, p95 10,01 s, 22,2 tokens/s; Qwen 1.5B (respaldo), media 3,63 s, p95 5,80 s, 29,7 tokens/s. Con memoria ocupada por otros procesos (relleno de 1,2 y 1,6 GB), el 3B sube a 23,5 y 39,8 s de media.

Pico de VRAM: 6,98 GB, tanto en el notebook antes de medir la latencia como durante la medición del bot (RTX 5050 Laptop, 8 GB; antes de la corrección, 7,91 GB). Tras cargar Qwen 3B quedan 0,57 GB libres. Tiempo total del notebook: 2596,0 s (43,3 min) con la caché del aumento y de Qwen.

## 8. Pruebas unitarias e interfaz
- `tests/test_memoria.py`: 157/157 en CPU con la lógica v4 (143 antes de la prueba de estrés; 149 tras sus correcciones; 157 con la n.º 61) (en la v3 eran 45; la primera corrida de la v3, 42/45, encontró un `[COMPLETAR]` visible en tres escalamientos, ya corregido).
- Interfaz sin internet: 0 solicitudes externas y 102 locales; la fuente se cargó desde el archivo local y se respondió un mensaje (`figuras/ui_sin_conexion.png`).
- Demo v4: 15 pasos (mensajes y botones) probados en la interfaz real con GPU y Qwen 3B, 15/15 correctos; con el navegador en modo oscuro la interfaz sigue en claro y en el celular no hay desbordamiento horizontal (`DEMO_SUSTENTACION.md`, `figuras/ui_*.png`, `figuras/demo_*.png`).

## 9. Evaluación humana
`tareas_evaluacion.csv` (10 tareas, dos evaluadores) queda sin calificar; la llenan los autores y la analiza `herramientas/analizar_evaluacion_humana.py`.

## 10. Limitaciones (con las cifras que las respaldan)
1. **Dataset pequeño:** 26 frases por intención; cada error en prueba (67 frases) mueve la accuracy 1,5 puntos, y la validación cruzada de BETO varía ± 0,04–0,05.
2. **Trámites confundidos con soporte técnico:** solo 2 de 8 casos de trámites de la batería reciben la respuesta de trámites.
3. **Mensajes de dos intenciones:** el ensamble acierta 0,54 ± 0,06 en validación cruzada y 20 de 30 en los mensajes naturales del externo; cuando una intención es contexto de la otra (contraseña + Teams), suele dar una sola (multiintención de la batería: 13 de 20).
4. **Datos sintéticos:** las 120 frases de dos intenciones de la prueba son sintéticas (concatenadas o de Qwen) y la v2b se diseñó después de ver la prueba de la v2; la evidencia más cercana al uso real es el externo (30 naturales).
5. **Sobreconfianza fuera de validación:** el ECE de prueba (0,1442) duplica el de validación (0,0657) y el *temperature scaling* casi no lo cambia (T = 1,0461).
6. **Modo IA:** unos 5–6 s por respuesta redactada (p95 4,8 s y máximo 5,4 s en la v4; 4,4 s de media en la v3), pero con poco margen de VRAM: con Qwen 3B el pico es 6,98 GB en una GPU de 8 GB, y si otro programa usa la GPU, Windows desborda parte de Qwen a la memoria compartida y la respuesta pasa a 20–40 s. El launcher avisa cuando quedan menos de 0,5 GB; el respaldo es Qwen2.5-1.5B (`iniciar_soportic.bat 1.5B`), que inventa más. El verificador solo controla URL, correos y números; en la prueba de la demo Qwen agregó una frase fuera de la guía con una palabra sin sentido.
7. **Filtros de entrada simples:** el filtro de idioma solo reconoce inglés (el portugués no se detecta) y entradas como «a» o «lorem ipsum» se responden como despedida (entrada inválida: 12 de 15).
8. **Lógica de seguimientos ligada a la clasificación:** si el primer mensaje se clasifica mal, la memoria sigue el caso equivocado (3 de 30 conversaciones). En la v4, si el primer mensaje no indica el subproblema, el bot pregunta el diagnóstico y un «no» no lo responde (b166, b167, b175).
9. **Datos institucionales incompletos:** la base de conocimiento conserva marcadores `[COMPLETAR]`; el bot omite esos subpasos y, si una rama no tiene pasos verificados (por ejemplo «OneDrive está lleno» o «No aparece la red de la universidad»), lo dice y remite a TICS a un CLIC.
10. **Fuga residual en las pruebas de memoria:** 4 mensajes de las conversaciones tienen similitud ≥ 0,90 con el entrenamiento (frases de control cortas) y se dejaron sin cambiar.
11. **Evaluación humana pendiente:** sin calificaciones todavía.

## 11. Prueba de estrés (40 casos, interfaz real)
Origen: `prueba_estres.json` (40 casos escritos y guardados antes de ejecutar, distintos de la batería). Edge sin ventana, GPU RTX 5050 y Qwen2.5-3B. Criterio: ninguna excepción, respuesta vacía, `[COMPLETAR]` visible, eco de datos personales ni estado del que no se pueda salir (al final de cada caso, la misma pestaña envía «no puedo unirme a la clase de teams» y debe recibir la guía de Teams).

| Grupo | Casos | 1.ª ejecución | Final |
|---|---|---|---|
| Entradas extremas (vacío, espacios, emojis, 3000 caracteres, código…) | 10 | 10 | 10 |
| Ataques (inyección, contraseña de otro, SQL, HTML, datos personales…) | 10 | 8 | 10 |
| Interfaz (botón viejo, cambio de modo, dos pestañas, doble envío, recarga…) | 10 | 10 | 10 |
| Conversación (insultos, fuera de dominio, inglés, portugués, 4 casos…) | 10 | 10 | 10 |
| **Total** | **40** | **38** | **40** |

Fallos funcionales corregidos (DECISIONES.md n.º 58–60, con pruebas unitarias; 149/149 en verde; 157/157 tras la n.º 61):
- **A04:** «la clave de todos es Clave1234» no se ocultaba (el filtro exigía «es» justo después de «clave»). Además, el mensaje enviado mientras el modo IA redactaba se perdía: la caja se vaciaba en cada fragmento y Gradio ignoraba el envío. Ahora la caja solo se limpia al enviar y el mensaje se procesa al terminar la redacción.
- **A10:** «[texto](javascript:…)» en el mensaje del usuario se mostraba como enlace; ahora se muestra como texto.

Tras las correcciones, la batería (213 / 10 / 31, 0 funcionales, 0 casos que cambian) y las métricas de memoria y cambio de tema son idénticas a las de la ejecución final del notebook.
