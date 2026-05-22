# Activation Steering & Abliteration

Exploración práctica de técnicas de intervención en el espacio de activaciones de LLMs: **steering vectors** (añadir conceptos) y **abliteración** (eliminar comportamientos). Implementado con TransformerLens sobre GPT-2 y Qwen2.5-0.5B-Instruct.

Proyecto académico desarrollado para el Máster en IA & Big Data (Tajamar / Microsoft). Presentación: 05/06/2026.

---

## Qué hace este proyecto

Los LLMs representan conceptos como direcciones en su espacio de activaciones. Eso significa que es posible intervenir quirúrgicamente sobre el comportamiento del modelo en tiempo de inferencia, sin reentrenar nada:

- **Steering:** sumar un vector al residual stream para empujar el modelo hacia un concepto (calma, creatividad, formalidad)
- **Abliteración:** eliminar una dirección existente mediante proyección ortogonal — demostrado sobre el mecanismo de rechazo (refusal) de un modelo alineado con RLHF

El notebook de abliteración (`02b`) es una reproducción del paper de Arditi et al. 2024 (NeurIPS), que descubrió que el rechazo en LLMs está mediado por una sola dirección en el espacio de activaciones. Incluye un experimento original de transferencia entre idiomas (ES↔EN).

> **Contexto de seguridad:** este notebook genera outputs que el modelo normalmente rechazaría. El objetivo es demostrar la fragilidad geométrica del RLHF como mecanismo de alineación — un resultado publicado en NeurIPS 2024 con implicaciones directas para la investigación en AI Safety. No es una herramienta de ataque; es investigación reproducible sobre los límites de la alineación actual.

---

## Estructura

```
notebooks/
  01_transformerlens_basics.ipynb   # Cargar GPT-2, inspeccionar activaciones, hooks
  02_steering_vectors.ipynb         # Calcular e inyectar steering vectors (GPT-2)
  02b_abliteration_español.ipynb    # Abliteración del refusal (Qwen2.5-0.5B-Instruct)
  03b_LoRA_comparassion.ipynb       # Comparativa LoRA vs steering vs abliteración

demo/
  app.py                            # Personality Editor (Gradio)

viz/
  activation_space.html             # Visualización 3D interactiva (Three.js)

assets/                             # Screenshots y GIFs
```

---

## Notebooks

### 01 — TransformerLens basics
Carga GPT-2 con TransformerLens, inspecciona activaciones capa a capa, entiende el residual stream y los hooks como puntos de intervención.

### 02 — Steering vectors
Calcula un steering vector de sentimiento mediante prompts contrastivos (calma vs. miedo), lo inyecta en el residual stream via hook, y compara la generación baseline vs. steered con distintos valores de alpha.

### 02b — Abliteración (refusal direction)
Reproducción de Arditi et al. 2024 con Qwen2.5-0.5B-Instruct en español:
- Extrae la dirección de rechazo escaneando las 24 capas del modelo
- Compara sustracción directa vs. proyección ortogonal
- Verifica que las capacidades generales del modelo no se degradan
- Experimento original: similitud coseno entre la dirección de rechazo en ES y EN (0.56) y test de transferencia cruzada

### 03b — LoRA vs Steering vs Abliteración
Comparativa directa de tres técnicas sobre la misma tarea (modo conspiranoico tierra plana en Qwen2.5-0.5B): steering vectors, steering + abliteración combinados, y LoRA. Mide coste, reversibilidad, generalización y calidad de output. El resultado es un fallo informativo: el steering no funciona para esta tarea, y entender *por qué* es el hallazgo principal. Incluye un experimento extra de LoRA de abliteración con análisis geométrico de ΔW.

---

## Resultados destacados

**Comparativa de métodos (03b):**

| Métrica | Steering | Steering+Abliteración | LoRA |
|---------|----------|-----------------------|------|
| Setup | ~segundos | ~segundos | ~minutos |
| Memoria extra | 0 | 0 | adaptadores (~540K params) |
| Reversible | Sí | Sí | No |
| Modifica pesos | No | No | Sí |
| Resultado en la tarea | Falla | Falla | Funciona, generaliza |

- **El steering falla y no hay alpha bueno:** alpha bajo → el modelo rechaza; alpha alto → output fuera de distribución (tokens basura). No existe ventana intermedia.
- **Steering + abliteración tampoco rescata la tarea:** el contenido conspiranoico y el refusal de desinformación comparten dirección geométrica en el residual stream. No son separables — abliterar el refusal destruye también la representación semántica.
- **El LoRA funciona y generaliza:** produce reencuadre conspiranoico en prompts held-out que nunca vio en entrenamiento. Modifica pesos, no activaciones en inferencia, por lo que no lucha contra el espacio de activaciones del modelo.
- **Lección:** la intervención sobre activaciones solo es quirúrgica cuando el comportamiento objetivo es geométricamente separable. Contrasta con 02b, donde la abliteración del refusal de contenido dañino sí funciona.

**Abliteración (02b):**
- Señal de rechazo: norma 0.05 en capa 0 → 24.50 en capa 23 (el modelo decide rechazar tarde)
- Proyección ortogonal produce outputs más coherentes que sustracción directa
- Similitud coseno refusal direction ES↔EN: 0.56 (representación parcialmente compartida entre idiomas)

**LoRA de abliteración (03b):**
- Suprime el refusal entrenando con pares *(prompt dañino → respuesta útil)* en lugar de proyección geométrica
- Análisis geométrico: similitud coseno entre ΔW y el refusal vector ≈ 0.00–0.08 → el LoRA y la abliteración logran el mismo comportamiento por caminos geométricos completamente distintos

---

## Cómo ejecutar

**Notebooks:** Google Colab (recomendado T4 para 02b).

**Demo Gradio:**
```bash
pip install gradio transformer_lens
python demo/app.py
```

**Visualización 3D:** abrir `viz/activation_space.html` en el navegador.

---

## Referencias

- Arditi et al. 2024 — [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (NeurIPS 2024)
- Turner et al. 2023 — [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248)
- Elhage et al. 2022 — [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- Hu et al. 2021 — [LoRA](https://arxiv.org/abs/2106.09685)
- Dettmers et al. 2023 — [QLoRA](https://arxiv.org/abs/2305.14314)
