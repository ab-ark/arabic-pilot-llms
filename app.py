import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1) Exact use-case names
USE_CASES = [
    "correct_rewriting",
    "correct_spelling",
    "correct_stylistic",
    "diacritics_output",
    "errorexplain_corrected",
    "format_output",
    "Grammar_output",
    "linguistic_output",
    "morphology_output",
    "namedentity",
    "Quranicquotation_output",
]

# 2) The model checkpoints
MODEL_IDS = {
    "ALLaM-7B":           "ALLaM-AI/ALLaM-7B-Instruct-preview",
    "SambaLingo-Arabic":  "sambanovasystems/SambaLingo-Arabic-Chat",
    "Mistral-7B-Arabic":  "malhajar/Mistral-7B-v0.1-arabic",
}

# 3) Prompt templates for each use case
PROMPT_TEMPLATES = {
    "correct_spelling": (
        "Please correct all spelling mistakes—both standard and typographical—in the following Arabic text:\n\n"
        "{text}"
    ),
    "Grammar_output": (
        "Correct errors in sentence structure, agreement, and syntax in the following Arabic text:\n\n"
        "{text}"
    ),
    "morphology_output": (
        "Analyze word formation and derivation errors and correct them in the following text:\n\n"
        "{text}"
    ),
    "linguistic_output": (
        "Detect semantic or contextual errors in the following text and correct them:\n\n"
        "{text}"
    ),
    "correct_stylistic": (
        "Improve phrasing and writing style for the following Arabic text:\n\n"
        "{text}"
    ),
    "format_output": (
        "Fix punctuation, spacing, and layout issues in the following Arabic text for better readability:\n\n"
        "{text}"
    ),
    "namedentity": (
        "Ensure proper spelling and usage of names and entities in the following Arabic text:\n\n"
        "{text}"
    ),
    "Quranicquotation_output": (
        "Verify and correct any Quranic quotations in the following Arabic text for accuracy and formatting:\n\n"
        "{text}"
    ),
    "diacritics_output": (
        "Add full or partial Arabic diacritical marks to improve pronunciation and clarity in the following text:\n\n"
        "{text}"
    ),
    "correct_rewriting": (
        "Rewrite the following Arabic text to improve clarity, flow, and style while preserving meaning:\n\n"
        "{text}"
    ),
    "errorexplain_corrected": (
        "Identify and correct all errors in the following Arabic text:\n\n"
        "{text}"
    ),
}

# 4) Caches for tokenizers and models
tokenizers = {}
models     = {}

def load_model(name):
    """
    Load tokenizer & model, cache them.
    If pad_token is missing, fall back to eos_token.
    """
    if name not in models:
        model_id = MODEL_IDS[name]
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
            mdl.resize_token_embeddings(len(tok))
        tokenizers[name] = tok
        models[name]     = mdl
    return tokenizers[name], models[name]

# 5) Default sampling‐based generation (ALLaM)
def generate_default(prompt, model_name):
    tok, mdl = load_model(model_name)
    inputs = tok(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(mdl.device)

    out = mdl.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    # decode only generated tokens
    input_len = inputs["input_ids"].shape[-1]
    gen_ids   = out[0, input_len:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()

# 6) Beam‐search generation for SambaLingo-Arabic
def generate_samba(prompt):
    tok, mdl = load_model("SambaLingo-Arabic")
    inputs = tok(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(mdl.device)

    # beam search, guarantee at least one new token
    out = mdl.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[-1] + 50,
        min_new_tokens=1,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    input_len = inputs["input_ids"].shape[-1]
    gen_ids   = out[0, input_len:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()

# 7) Beam‐search generation for Mistral-7B-Arabic
def generate_mistral(prompt):
    tok, mdl = load_model("Mistral-7B-Arabic")
    inputs = tok(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(mdl.device)

    out = mdl.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[-1] + 50,
        min_new_tokens=1,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    input_len = inputs["input_ids"].shape[-1]
    gen_ids   = out[0, input_len:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()

# 8) Main pipeline dispatcher
def run_pipeline(text, use_case, model_name):
    template = PROMPT_TEMPLATES.get(use_case, "{text}")
    prompt   = template.format(text=text)

    if model_name == "SambaLingo-Arabic":
        return generate_samba(prompt)
    elif model_name == "Mistral-7B-Arabic":
        return generate_mistral(prompt)
    else:
        return generate_default(prompt, model_name)

# 9) Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Arabic NLP Playground")
    with gr.Row():
        txt_input = gr.Textbox(
            label="Enter your Arabic text here",
            placeholder="Type or paste Arabic text…",
            lines=5
        )
    with gr.Row():
        uc_dropdown    = gr.Dropdown(choices=USE_CASES,          label="Select Use Case")
        model_dropdown = gr.Dropdown(choices=list(MODEL_IDS.keys()), label="Select Model")
    run_btn    = gr.Button("Run")
    output_box = gr.Textbox(label="Model Output", lines=4)

    run_btn.click(
        fn=run_pipeline,
        inputs=[txt_input, uc_dropdown, model_dropdown],
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()
