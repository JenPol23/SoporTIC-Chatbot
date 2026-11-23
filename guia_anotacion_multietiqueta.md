# Guía de anotación multietiqueta — SoporTIC (Fase 3)

Definida el 2026-11-14, antes de entrenar o evaluar cualquier modelo multietiqueta. Se aplica por igual a todas las particiones (entrenamiento, validación, prueba) y al conjunto de generalización, sin mirar predicciones de ningún modelo.

## Reglas
1. **Dos problemas distintos → dos etiquetas.** Si la frase describe dos problemas de soporte que necesitan guías distintas (p. ej., «olvidé la clave y además no puedo entregar en Moodle»), lleva las dos intenciones.
2. **Servicio como contexto → una etiqueta.** Si otro servicio aparece solo como contexto del problema (dónde ocurre, con qué cuenta, para qué se necesita, el plazo o la causa), lleva solo la intención del problema. Ejemplos: «la clave para entrar a moodle», «en el pc de la sala no abre el correo», «mañana entrego y colab no me da gpu».
3. **Plazos y urgencia no son problemas.** «mañana entrego», «el parcial es en una hora» no agregan `moodle_entregas`.
4. **Las definiciones aprobadas en la Fase 1 siguen vigentes.** Si la definición de una intención ya cubre el caso (p. ej., «el computador de la sala no tiene internet» es `salas_computo`), no se agrega otra etiqueta.
5. **Intenciones de conversación.** Un saludo o un agradecimiento que acompaña un problema no se etiqueta aparte en el dataset (las respuestas fijas los cubren); el conjunto externo conserva las etiquetas de sus autores.
6. **Máximo tres etiquetas** por frase.

## Formato de los archivos multietiqueta
`dataset_soporte_multietiqueta.csv`: `id, texto, intenciones, origen, particion`, donde `id` es la fila de `dataset_soporte.csv`, `intenciones` separa las etiquetas con `|` y `particion` (`train`, `val`, `test`) reproduce la división estratificada 70/15/15 de la v1 (semilla 42).
`frases_generalizacion_multietiqueta.csv`: `id, texto, intenciones, origen`.
