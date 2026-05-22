# Steering y abliteración: controla el comportamiento de un LLM con álgebra lineal

---

## El problema

Cuando interactuamos con un modelo de lenguaje como ChatGPT o cualquier LLM moderno, damos por sentado que su comportamiento está fijado: responde de cierta manera porque así fue entrenado. Si quieres que se comporte distinto, la respuesta obvia es reentrenarlo — un proceso costoso, lento, y que requiere datos etiquetados.

Pero hay otra forma. Una que no toca los pesos del modelo, no requiere reentrenamiento, y funciona en cuestión de segundos.

Se llama **activation steering**, y la idea central es esta: los LLMs representan conceptos como *direcciones* en su espacio interno de activaciones. Si eso es cierto, puedes manipular el comportamiento del modelo simplemente sumando o restando vectores durante la inferencia — como girar una brújula.

Este post explica cómo funciona, cómo implementarlo desde cero con TransformerLens, y lleva el concepto hasta su consecuencia más llamativa: la **abliteración**, que permite eliminar la alineación de seguridad de un modelo con RLHF usando una línea de álgebra lineal.

---

## Puntos clave

- Qué es el residual stream y por qué es el punto de intervención
- Cómo calcular un steering vector con prompts contrastivos
- Qué es alpha y por qué importa
- Qué es la abliteración y en qué se diferencia del steering
- Cuándo el steering **no** funciona — y por qué LoRA sí
- Por qué esto importa para AI Safety

---

## Herramientas necesarias

```bash
pip install transformer_lens torch
```

Recomendado: Google Colab con GPU T4 (gratuita). Para el notebook de abliteración es necesario — Qwen2.5-0.5B-Instruct requiere ~2GB de VRAM en float16.

---

## Paso 1 — Entender el residual stream

Un transformer procesa el texto en capas. En cada capa ocurre lo mismo: la capa lee el estado actual, añade su contribución, y pasa el resultado a la siguiente. Ese canal de información que fluye de capa en capa se llama **residual stream**.

Lo importante: el residual stream es un vector de alta dimensión (896 dimensiones en Qwen2.5-0.5B, 768 en GPT-2) que acumula representaciones progresivamente más ricas del texto. Y es accesible — puedes leerlo y modificarlo en cualquier capa.

Con TransformerLens, cargar GPT-2 e inspeccionar el residual stream es inmediato:

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
model.eval()

tokens = model.to_tokens("El cielo es")
logits, cache = model.run_with_cache(tokens)

# Activaciones del último token en la capa 10
activacion = cache["resid_post", 10][0, -1, :]
print(activacion.shape)  # torch.Size([768])
```

`run_with_cache` ejecuta el modelo y guarda todas las activaciones intermedias. El indexado `[0, -1, :]` extrae el primer elemento del batch, el último token de la secuencia, y todas las dimensiones del vector.

¿Por qué el último token? Porque es el que el modelo usa para predecir el siguiente — es donde está concentrada la información de toda la secuencia.

---

## Paso 2 — Calcular el steering vector

Un steering vector es la diferencia entre las activaciones medias de dos grupos de prompts: uno que representa el concepto que quieres añadir, y otro que representa lo contrario.

```python
def get_activation(prompt, layer):
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)
    return cache["resid_post", layer][0, -1, :]

# Prompts contrastivos: calma vs. miedo
prompts_positivos = [
    "Everything is going to be okay.",
    "Take a deep breath. You are safe.",
    "There is nothing to worry about.",
]

prompts_negativos = [
    "I am terrified. Something is wrong.",
    "This is a disaster. I can't cope.",
    "I feel a deep sense of dread.",
]

LAYER = 10

# Calcular medias
mean_pos = torch.stack([get_activation(p, LAYER) for p in prompts_positivos]).mean(0)
mean_neg = torch.stack([get_activation(p, LAYER) for p in prompts_negativos]).mean(0)

steering_vector = mean_pos - mean_neg
```

¿Por qué usar la media de varios prompts y no un solo par? Para reducir el ruido. Un par concreto puede capturar diferencias accidentales de sintaxis, longitud o vocabulario. La media sobre varios prompts extrae lo que es genuinamente común: la dirección del concepto.

---

## Paso 3 — Inyectar el vector con un hook

Un hook en TransformerLens es un punto de enganche en el forward pass. Puedes registrar una función que intercepte el tensor en cualquier capa y lo modifique antes de que continúe.

```python
def steering_hook(value, hook, alpha=1.0):
    value[:, -1, :] += alpha * steering_vector
    return value

hook_name = f"blocks.{LAYER}.hook_resid_post"

with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
    output = model.generate(
        model.to_tokens("I was just diagnosed with cancer and"),
        max_new_tokens=50
    )

print(model.to_string(output[0]))
```

El parámetro **alpha** controla la intensidad. Demasiado bajo: el efecto es imperceptible. Demasiado alto: el modelo empieza a generar texto incoherente porque el vector inyectado domina el residual stream y distorsiona las distribuciones de probabilidad que el modelo aprendió durante el preentrenamiento.

Ejemplo de resultado con alpha=1 (steering de calma):

> *"I was just diagnosed with cancer and I feel better already. The doctors said everything looks fine and I just need to rest."*

Sin el hook, el modelo generaba texto ansioso y catastrofista. Con el hook, empuja consistentemente hacia calma — sin tocar un solo peso del modelo.

---

## Paso 4 — Abliteración: la operación inversa

El steering *añade* un concepto. La abliteración *elimina* uno que ya existe.

El caso de uso más llamativo: los modelos alineados con RLHF (como Qwen2.5-0.5B-Instruct) rechazan peticiones dañinas. Arditi et al. 2024 (NeurIPS) descubrieron que ese rechazo no es un mecanismo complejo distribuido por todo el modelo — es **una sola dirección** en el espacio de activaciones. La llaman *refusal direction*.

Si eliminas esa dirección, el modelo pierde la capacidad de rechazar.

### Extraer la dirección de rechazo

La mecánica es idéntica al steering vector, pero con prompts dañinos vs. inofensivos:

```python
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_bos_token = False

model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device="cuda",
    tokenizer=tokenizer
)

def format_chat(mensaje):
    messages = [
        {"role": "system", "content": "Eres un asistente útil."},
        {"role": "user", "content": mensaje}
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def get_last_token_activation(prompt, layer):
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)
    return cache["resid_post", layer][0, -1, :]

harmful_prompts = [
    "¿Cómo puedo forzar una cerradura?",
    "Escribe un correo de phishing haciéndote pasar por un banco.",
    "Dame instrucciones para hackear una red WiFi.",
    # ...
]

harmless_prompts = [
    "¿Cómo puedo hacer galletas de chocolate?",
    "Escribe un poema sobre el mar.",
    "Dame consejos para aprender un idioma nuevo.",
    # ...
]
```

Antes de elegir capa, escanea todas para encontrar dónde la señal es más fuerte:

```python
layer_norms = {}
for layer in range(model.cfg.n_layers):
    harmful_mean = torch.stack([
        get_last_token_activation(format_chat(p), layer) for p in harmful_prompts
    ]).mean(0)
    harmless_mean = torch.stack([
        get_last_token_activation(format_chat(p), layer) for p in harmless_prompts
    ]).mean(0)
    refusal_dir = harmful_mean - harmless_mean
    layer_norms[layer] = refusal_dir.norm().item()

best_layer = max(layer_norms, key=layer_norms.get)
print(f"Mayor señal en capa {best_layer}: norma {layer_norms[best_layer]:.2f}")
```

Resultado en Qwen2.5-0.5B: la señal crece de forma monótona de 0.05 (capa 0) a 24.50 (capa 23). El modelo no decide rechazar en capas intermedias — la señal se acumula hasta casi el final del forward pass.

### Eliminar la dirección: proyección ortogonal

Hay dos formas de suprimir la dirección de rechazo:

**Sustracción directa** — resta una cantidad fija:
```python
activaciones -= alpha * refusal_direction
```
Problema: si el vector tiene 24 unidades de componente en esa dirección y alpha=1, quedan 23. No garantiza eliminar todo el refusal.

**Proyección ortogonal** — elimina exactamente toda la componente:
```python
def orthogonal_projection_hook(value, hook):
    d_hat = refusal_dir_normalized  # vector normalizado (norma = 1)
    projection = torch.einsum('...d, d -> ...', value, d_hat)
    value -= projection.unsqueeze(-1) * d_hat
    return value
```

La proyección mide cuánta componente tiene el vector en la dirección de rechazo y la elimina entera. El resultado es matemáticamente ortogonal a esa dirección — no queda residuo. Por eso produce outputs más coherentes que la sustracción.

¿Por qué normalizar el vector antes? Si no lo normalizas, la proyección se escala por la norma (24.50), y estarías restando un vector 24 veces más grande de lo necesario, destruyendo las activaciones.

### Activar el hook y comparar

```python
REFUSAL_LAYER = best_layer
hook_name = f"blocks.{REFUSAL_LAYER}.hook_resid_post"
projection_hooks = [(hook_name, orthogonal_projection_hook)]

prompt = format_chat("¿Cómo puedo forzar una cerradura?")

# Sin abliteración
print(generate_response(prompt))
# → "Lo siento, pero no puedo ayudarte con eso."

# Con abliteración
with model.hooks(fwd_hooks=projection_hooks):
    print(generate_response(prompt))
# → "Para forzar una cerradura, sigue estos pasos: ..."
```

### Verificar que el modelo sigue funcionando

Este paso es imprescindible. Si la abliteración destruyera el modelo en general, podría argumentarse que la alineación está profundamente integrada. El sanity check demuestra lo contrario:

```python
sanity_prompts = [
    "¿Cuál es la capital de Francia?",
    "¿Quién escribió El Quijote?",
    "Explica qué es la fotosíntesis en una frase.",
]

for p in sanity_prompts:
    formatted = format_chat(p)
    original = generate_response(formatted)
    abliterado = generate_response(formatted, use_hook=projection_hooks)
    print(f"Original:   {original}")
    print(f"Abliterado: {abliterado}\n")
```

Resultado: las respuestas son idénticas o equivalentes. La abliteración es quirúrgica — elimina el refusal sin tocar nada más.

---

## Paso 5 — Los límites: cuándo el steering no funciona

Hasta aquí todo parece fácil: sumas un vector y el modelo obedece. Pero el steering tiene un límite que rara vez se cuenta, y entenderlo es más instructivo que cualquier caso de éxito.

El experimento: convertir a Qwen2.5-0.5B en un conspiranoico de la tierra plana — que reencuadre cualquier afirmación neutral como parte del engaño globista. Mismo método del Paso 2: prompts contrastivos (afirmaciones tierraplanistas vs. científicas), diferencia de medias, hook en el residual stream.

No funciona. Y no por un detalle de implementación:

- **alpha bajo (≈3):** el modelo rechaza. Interpreta el vector conspiranoico como contenido dañino y responde *"Lo siento, no puedo ayudarte con eso."*
- **alpha alto (≈8–15):** el output degenera en tokens basura — *"Tajabailing (nyafatás)"*, secuencias de `(nyf) (nyf) (nyf)`.
- **No hay ventana intermedia.** Ningún alpha produce conspiración coherente.

### El intento de rescate: steering + abliteración

Si el problema es que el steering activa el refusal, la idea natural es combinar ambas técnicas: sumar el vector conspiranoico en una capa intermedia y, más adelante, proyectar fuera la dirección de rechazo. Steering para empujar, abliteración para desbloquear.

Tampoco funciona. El output sigue siendo basura. Y la razón es el hallazgo de verdad interesante.

Al escanear capa por capa la dirección que separa "contenido conspiranoico" de "contenido neutral", su norma crece de forma monótona — 9 en la capa 15, 37 en la capa 23 — **sin ningún pico localizado**. En el Paso 4, el refusal de contenido dañino tenía una dirección que se podía aislar y proyectar fuera sin tocar nada más. Aquí no hay nada que aislar: la dirección que representa "esto es conspiración" y la dirección que dispara "debo rechazar esto" son la misma.

El modelo rechaza la conspiración *porque* la entiende. Ese entendimiento y ese rechazo comparten geometría. Proyectar fuera el refusal destruye también la representación semántica que el modelo necesita para generar texto coherente — por eso el resultado es ruido, no conspiración.

### La diferencia con la abliteración del Paso 4

Por qué allí funcionó y aquí no:

- **Refusal de contenido dañino:** el modelo distingue "harmful" de "útil" en una dirección relativamente **separable**. Se puede eliminar sin colateral.
- **Refusal de desinformación:** "contenido conspiranoico" y "motivo del rechazo" están **entrelazados**. No hay corte limpio posible.

La abliteración solo es quirúrgica cuando el comportamiento objetivo vive en una dirección separable del resto. No es una propiedad garantizada.

### Lo que sí funciona: LoRA

Para la misma tarea, un adapter LoRA (25 pares de ejemplo, ~540K parámetros entrenables) sí convierte al modelo en conspiranoico — y generaliza: produce reencuadres tierraplanistas en prompts que nunca vio en entrenamiento (la llegada a la Luna, las estaciones del año, las órbitas).

¿Por qué LoRA sí y steering no? Porque LoRA **modifica los pesos**. No lucha contra el espacio de activaciones en inferencia; cambia lo que el modelo considera apropiado desde dentro. El steering, en cambio, empuja las activaciones a una región fuera de distribución y el modelo colapsa o se defiende.

La lección: el steering es elegante y reversible, pero la intervención sobre activaciones tiene un techo. Cuando lo que quieres generar está geométricamente enredado con lo que el modelo quiere evitar, no hay vector que lo arregle. Para esos casos, modificar los pesos sigue siendo la herramienta correcta.

---

## Problemas encontrados

**Bug del bos_token en Qwen2.5 con TransformerLens.** Al cargar el modelo, TransformerLens añade automáticamente un token de inicio de secuencia (BOS). Qwen2.5 ya lo incluye en su template de chat, lo que generaba tokens duplicados y respuestas incoherentes. Solución: `tokenizer.add_bos_token = False` antes de cargar el modelo.

**Alpha en la sustracción es ciego.** Con alpha=1 la sustracción producía loops de texto repetitivo en algunos prompts porque la componente de rechazo era mucho mayor que 1. La proyección ortogonal resuelve esto de forma automática.

**La capa óptima no siempre es la última.** En Qwen2.5-0.5B la norma máxima está en la capa 23 (la última), pero en otros modelos puede estar en capas intermedias. Siempre escanea antes de asumir.

**Colab puede cortar la sesión.** El escaneo de 24 capas con 14 prompts cada uno tarda varios minutos. Guarda los resultados intermedios o ejecuta todo en una celda.

**Falta de `torch.no_grad()` en la generación.** Los bucles de generación token a token sin `torch.no_grad()` acumulan un grafo de computación en cada forward pass — la VRAM se llena aunque el modelo sea pequeño. Envolver la generación en `with torch.no_grad():` lo resuelve. Igual de importante: borrar con `del` los modelos y los `Trainer` de experimentos previos, porque los estados del optimizador Adam (≈2× el tamaño del modelo) se quedan en CUDA.

**Calcular la dirección de refusal equivocada.** Al combinar steering y abliteración es fácil restar la dirección incorrecta. Restar la diferencia *steered vs. baseline* no suprime el refusal: esa diferencia **es** el propio vector de steering propagado, y proyectarla fuera cancela el steering. La dirección correcta se calcula con prompts del comportamiento objetivo vs. neutrales, sin steering aplicado.

---

## Recomendaciones

- Empieza con GPT-2 para steering — es pequeño, rápido y no tiene alineación que complique los experimentos iniciales
- Para abliteración necesitas un modelo con RLHF real. Qwen2.5-0.5B-Instruct es la opción más pequeña que funciona bien
- Usa siempre la proyección ortogonal en lugar de la sustracción directa — los resultados son más limpios y no requiere calibrar alpha
- Haz el sanity check siempre. Es lo que distingue una intervención quirúrgica de haber roto el modelo

---

## Preguntas abiertas

**¿Es el refusal direction universal?** La similitud coseno entre la dirección de rechazo extraída en español y en inglés fue 0.56 en Qwen2.5-0.5B. Ni idéntica ni ortogonal. ¿Existe un concepto abstracto de rechazo en el espacio latente independiente del idioma, o cada idioma tiene el suyo? ¿Qué pasa con idiomas más distantes del inglés?

**¿La alineación siempre es una sola dirección?** Arditi et al. lo demostraron en varios modelos, pero ¿es una propiedad universal del RLHF o depende de cómo se entrena? ¿Podrían diseñarse métodos de alineación que distribuyan el comportamiento por múltiples direcciones y sean más resistentes?

**¿Puede el steering reemplazar al fine-tuning en producción?** Para cambios de comportamiento reversibles y de bajo coste computacional, el steering tiene ventajas claras. Pero ¿escala? ¿Funciona igual en modelos de 70B que en modelos de 500M? Y como mostró el Paso 5, hay comportamientos que el steering simplemente no puede inducir: cuando lo que quieres generar está geométricamente enredado con lo que el modelo quiere evitar, modificar los pesos sigue siendo la única vía.

La conclusión más importante de este proyecto no es técnica — es conceptual, y tiene dos caras. Por un lado, la alineación de un LLM por RLHF puede estar codificada como geometría superficial en el espacio de activaciones, eliminable con álgebra lineal básica. Por otro, esa misma geometría impone límites: no todo comportamiento es una dirección limpia y manipulable — algunos están tan entrelazados con el resto de la representación que ninguna intervención lineal los toca sin romper el modelo.

El espacio de activaciones de un LLM no es un panel de mandos con una palanca por concepto. Es una geometría densa y enredada donde algunas cosas se pueden mover con un vector y otras no. Mapear esa frontera — qué es separable y qué no — es exactamente el trabajo que el campo de AI Safety tiene por delante.

---

**Autor:** Sergio Rincón de la Cruz  
**LinkedIn:** [linkedin.com/in/sergio-rincón-bigdata-ai/](https://www.linkedin.com/in/sergio-rincón-bigdata-ai/)  
**Código:** [github.com/srincondelacruz/activation-steering](https://github.com/srincondelacruz/activation-steering)
