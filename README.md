# Activation Steering — GPT-2

Proyecto de exploración de **steering vectors** como técnica de control de comportamiento en LLMs. Compara activation steering con LoRA e incluye una interfaz interactiva y una visualización 3D del espacio de activaciones.

**Presentación:** 05/06/2026

---

## Estructura

```
notebooks/
  01_transformerlens_basics.ipynb   # Cargar GPT-2 e inspeccionar activaciones
  02_steering_vectors.ipynb         # Calcular e inyectar steering vectors
  03_lora_comparison.ipynb          # Comparativa LoRA vs steering

demo/
  app.py                            # Personality Editor (Gradio)

viz/
  activation_space.html             # Visualización 3D interactiva (Three.js)

assets/                             # Screenshots y GIFs
```

## Resultados

| Métrica | Steering | LoRA |
|---------|----------|------|
| Setup | ~segundos | ~minutos |
| Memoria extra | 0 | adaptadores |
| Reversible | Sí | No |
| Efectos secundarios | — | — |

## Cómo ejecutar

**Notebooks:** abrir en Google Colab con la extensión de VS Code o directamente en colab.research.google.com.

**Demo Gradio:**
```bash
pip install gradio transformer_lens
python demo/app.py
```

**Visualización 3D:** abrir `viz/activation_space.html` en el navegador.

## Referencias

- Turner et al. 2023 — [Activation Addition](https://arxiv.org/abs/2308.10248)
- Elhage et al. 2022 — [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- Hu et al. 2021 — [LoRA](https://arxiv.org/abs/2106.09685)
- Dettmers et al. 2023 — [QLoRA](https://arxiv.org/abs/2305.14314)
