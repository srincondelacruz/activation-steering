# Vector Steering — Plan de ejecución (3 semanas)

**Fecha de presentación:** 05/06/2025
**Fecha de inicio:** 14/05/2025

---

## SEMANA 1 — Fundamentos (14–20 mayo)

**Objetivo:** entender los papers, tener TransformerLens funcionando, y primer steering vector operativo.

### Día 1–2 (miércoles 14 – jueves 15)
**Lectura de papers — solo abstract + intro + conclusiones**

- [ ] Turner et al. 2023 — Activation Steering (el más importante)
  - Foco: qué es ActAdd, cómo se computan los steering vectors, en qué capa se inyectan
  - Link: https://arxiv.org/abs/2308.10248
- [ ] Elhage et al. 2022 — Toy Models of Superposition
  - Foco: por qué los modelos representan más features que dimensiones, concepto de superposición
  - Link: https://transformer-circuits.pub/2022/toy_model/index.html (versión web)
- [ ] Hu et al. 2021 — LoRA
  - Foco: descomposición de bajo rango, por qué ΔW tiene rango bajo, matrices A y B
  - Link: https://arxiv.org/abs/2106.09685
- [ ] Dettmers et al. 2023 — QLoRA
  - Foco: NF4, double quantization, cómo encaja con LoRA
  - Link: https://arxiv.org/abs/2305.14314

**Entregable del día 2:** un documento personal de 1–2 páginas con lo que has entendido de cada paper en tus propias palabras. No es para entregar — es para verificar que lo entiendes.

### Día 3–4 (viernes 16 – sábado 17)
**Hands-on con TransformerLens**

- [ ] Instalar TransformerLens en Google Colab
- [ ] Cargar GPT-2 y ejecutar una inferencia básica
- [ ] Inspeccionar las activaciones de una capa intermedia
- [ ] Entender la estructura de los hooks (cómo interceptar y modificar activaciones)
- [ ] Ejecutar un ejemplo sencillo del repo de TransformerLens

**Entregable del día 4:** un notebook limpio que carga GPT-2, pasa un prompt, y muestra las activaciones de la capa 10 como tensor con su shape. Nada más.

### Día 5–6 (domingo 18 – lunes 19)
**Primer steering vector funcional**

- [ ] Definir dos prompts contrastivos (ej. "I am certain that" vs "I doubt that")
- [ ] Extraer activaciones de ambos en una capa intermedia (probar capas 8–14)
- [ ] Calcular el steering vector: diferencia de las activaciones medias
- [ ] Inyectar el vector durante la inferencia con un hook
- [ ] Comparar outputs con y sin steering (mínimo 5 prompts diferentes)
- [ ] Documentar qué capa y qué escala (alpha) dan mejores resultados

**Entregable del día 6:** notebook funcional con steering vector de escepticismo inyectado en GPT-2. Output antes/después visible.

### Día 7 (martes 20)
**Consolidación y revisión**

- [ ] Releer las secciones técnicas de Turner et al. que ahora ya puedes entender con contexto práctico
- [ ] Anotar preguntas pendientes
- [ ] Limpiar el notebook — comentarios claros, celdas ordenadas
- [ ] Commit inicial a GitHub (repo nuevo, README básico)

**Checkpoint semana 1:** tienes un notebook que funciona, entiendes los papers, y puedes explicar qué hace cada línea de código. Si no estás aquí el martes 20, la semana 2 se simplifica.

---

## SEMANA 2 — Proyecto completo (21–27 mayo)

**Objetivo:** añadir la comparativa con LoRA, montar la interfaz Gradio, y tener el esqueleto de la charla cerrado.

### Día 8–9 (miércoles 21 – jueves 22)
**Comparativa LoRA vs Steering**

- [ ] Elegir una tarea concreta para comparar (ej. reducir toxicidad o cambiar sentimiento)
- [ ] Implementar la misma modificación de comportamiento con steering (ya lo tienes)
- [ ] Implementar la misma modificación con LoRA usando PEFT de Hugging Face
  - Mismo modelo base (GPT-2)
  - Dataset pequeño para fine-tuning (unas pocas docenas de ejemplos)
  - Entrenar con QLoRA si la T4 lo permite
- [ ] Definir métricas de comparación:
  - Calidad del output (coherencia, relevancia)
  - Coste computacional (tiempo, memoria)
  - Reversibilidad (steering se quita al instante, LoRA no)
  - Efectos secundarios (¿el modelo se degrada en otras tareas?)

**Entregable del día 9:** tabla comparativa con resultados reales, no teóricos.

### Día 10–11 (viernes 23 – sábado 24)
**Interfaz Gradio (Personality Editor)**

- [ ] Instalar Gradio en Colab
- [ ] Crear interfaz con:
  - Input de texto (prompt del usuario)
  - Dropdown o botones para elegir personalidad (escepticismo, creatividad, formalidad)
  - Slider de intensidad (alpha del steering vector)
  - Dos outputs: baseline vs steered
- [ ] Testear con 10+ prompts variados
- [ ] Capturar GIFs/screenshots para el README

**Entregable del día 11:** interfaz Gradio funcional, capturas guardadas.

### Día 12–13 (domingo 25 – lunes 26)
**Esqueleto técnico de la charla**

- [ ] Escribir el guión de cada acto:
  - Acto 1: Hook — qué voy a decir literalmente en los primeros 2 minutos
  - Acto 2: Activation Steering — qué conceptos explico, en qué orden, con qué ejemplos
  - Acto 3: LoRA/QLoRA — qué muestro de la comparativa
  - Acto 4: Ortogonalización — cómo lo conecto con los dos anteriores
  - Acto 5: Demo — qué hago en vivo, plan B si falla la demo
- [ ] Identificar los momentos donde necesito una visualización
- [ ] Definir qué va en slides vs qué va en demo en vivo

**Entregable del día 13:** guión completo de la charla en markdown, con timing por acto.

### Día 14 (martes 27)
**Consolidación semana 2**

- [ ] Limpiar todo el notebook — un solo notebook final o separar en módulos claros
- [ ] Actualizar GitHub: README con descripción del proyecto, GIFs, tabla de resultados
- [ ] Verificar que todo corre desde cero en Colab (reiniciar runtime y ejecutar)

**Checkpoint semana 2:** tienes el proyecto técnico completo. Lo que queda es presentación.

---

## SEMANA 3 — Presentación (28 mayo – 4 junio)

**Objetivo:** slides, visualización 3D, ensayo, y pulido final.

### Día 15–16 (miércoles 28 – jueves 29)
**Visualización 3D interactiva**

- [ ] Construir la visualización con Three.js o Plotly 3D:
  - Espacio de activaciones con vectores conceptuales (puntos/flechas en 3D)
  - Efecto del steering: punto que se desplaza cuando inyectas el vector
  - Subespacio LoRA como plano semitransparente de bajo rango
- [ ] Debe funcionar en navegador (HTML standalone o dentro de un artifact)
- [ ] Interactiva: el público puede rotar, hacer zoom, activar/desactivar vectores

**Entregable del día 16:** visualización 3D funcional en HTML.

### Día 17–18 (viernes 30 – sábado 31)
**Slides**

- [ ] Crear slides siguiendo el guión de la semana 2
- [ ] Reglas:
  - Máximo 20–25 slides para 1 hora
  - Poco texto, mucho visual
  - Cada slide tiene un solo mensaje
  - Código en slides solo si es 5 líneas o menos — el resto se muestra en vivo
- [ ] Incluir la visualización 3D como slide interactiva o link
- [ ] Incluir capturas del Personality Editor

**Entregable del día 18:** deck completo.

### Día 19–20 (domingo 1 – lunes 2 junio)
**Ensayo**

- [ ] Ensayar la charla completa con timer — MÍNIMO 2 veces
- [ ] Primer ensayo: solo, midiendo tiempos por acto
- [ ] Segundo ensayo: grabarte (aunque sea audio) y escucharte
- [ ] Ajustar timing — si te pasas, recortar el Acto 4 (ortogonalización) primero
- [ ] Preparar plan B para la demo:
  - Si Colab falla → tener outputs pre-generados en las slides
  - Si la visualización falla → tener screenshots estáticos

**Entregable del día 20:** charla ensayada, timing verificado, plan B preparado.

### Día 21 (martes 3 junio)
**Pulido final**

- [ ] Revisar todo el código una última vez
- [ ] Verificar que la demo corre en Colab desde cero
- [ ] Verificar que la visualización 3D carga sin errores
- [ ] Subir versión final a GitHub con README pulido
- [ ] Preparar el post de LinkedIn (borrador)

### Día 22 (miércoles 4 junio)
**Buffer**

Día libre de emergencia. Si todo va bien, descanso. Si algo se rompió, se arregla aquí.

### Día 23 (jueves 5 junio)
**Presentación**

---

## Plan de recorte (si vas mal de tiempo)

| Prioridad | Qué recortar | Impacto |
|-----------|--------------|---------|
| Primero | Comparativa LoRA (semana 2, días 8–9) | Bajo — la charla funciona sin ella |
| Segundo | Interfaz Gradio (semana 2, días 10–11) | Medio — pierdes el "wow" visual pero la demo en notebook basta |
| Tercero | Visualización 3D (semana 3, días 15–16) | Medio — sustituyes con diagrama estático en slides |
| NUNCA | El notebook de steering | Sin esto no hay proyecto |
| NUNCA | El esqueleto de la charla | Sin esto no hay presentación |

---

## Checklist final pre-presentación

- [ ] Notebook corre desde cero en Colab sin errores
- [ ] Repo de GitHub con README, código limpio, y capturas/GIFs
- [ ] Slides completas con timing verificado
- [ ] Demo en vivo testeada + plan B con outputs pre-generados
- [ ] Visualización 3D funcional (o screenshots estáticos como fallback)
- [ ] Charla ensayada mínimo 2 veces
- [ ] Post de LinkedIn en borrador
