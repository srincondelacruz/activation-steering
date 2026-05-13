import torch
import gradio as gr
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Modelo y steering vectors
# ---------------------------------------------------------------------------

model = HookedTransformer.from_pretrained("gpt2")
model.eval()

LAYER = 10
PERSONALITIES = {
    "Escepticismo": {
        "positive": ["I am certain that", "It is absolutely true that", "Without any doubt,"],
        "negative": ["I doubt that", "I am not sure whether", "It is unclear whether"],
    },
    "Creatividad": {
        "positive": ["The conventional approach is", "The standard method is", "Traditionally,"],
        "negative": ["Imagine if", "What if we instead", "A creative alternative would be"],
    },
    "Formalidad": {
        "positive": ["Hey, so basically", "It's like, you know,", "Honestly,"],
        "negative": ["It is hereby stated that", "In accordance with", "The aforementioned"],
    },
}


def compute_steering_vector(personality_key):
    cfg = PERSONALITIES[personality_key]
    def mean_act(prompts):
        acts = []
        for p in prompts:
            tokens = model.to_tokens(p)
            _, cache = model.run_with_cache(tokens)
            acts.append(cache["resid_post", LAYER][0, -1, :])
        return torch.stack(acts).mean(0)
    return mean_act(cfg["negative"]) - mean_act(cfg["positive"])


steering_vectors = {k: compute_steering_vector(k) for k in PERSONALITIES}


# ---------------------------------------------------------------------------
# Generación
# ---------------------------------------------------------------------------

def generate(prompt, personality, alpha, max_new_tokens=40):
    tokens = model.to_tokens(prompt)
    sv = steering_vectors[personality]

    def hook_fn(value, hook):
        value[:, -1, :] += alpha * sv
        return value

    baseline = model.generate(model.to_tokens(prompt), max_new_tokens=max_new_tokens)
    steered = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        hooks=[(f"blocks.{LAYER}.hook_resid_post", hook_fn)],
    )
    return model.to_string(baseline[0]), model.to_string(steered[0])


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------

with gr.Blocks(title="Personality Editor") as demo:
    gr.Markdown("# Personality Editor\nModifica el comportamiento de GPT-2 con steering vectors.")

    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt", placeholder="Scientists have discovered that")
        personality_dd = gr.Dropdown(choices=list(PERSONALITIES.keys()), value="Escepticismo", label="Personalidad")

    alpha_slider = gr.Slider(minimum=0, maximum=30, value=15, step=1, label="Intensidad (alpha)")

    generate_btn = gr.Button("Generar")

    with gr.Row():
        baseline_out = gr.Textbox(label="Baseline (sin steering)")
        steered_out = gr.Textbox(label="Steered")

    generate_btn.click(
        fn=generate,
        inputs=[prompt_input, personality_dd, alpha_slider],
        outputs=[baseline_out, steered_out],
    )

if __name__ == "__main__":
    demo.launch()
