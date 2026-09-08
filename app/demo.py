"""Gradio UI for Intent Classification Engine."""
from __future__ import annotations
from pathlib import Path
import gradio as gr
import joblib
from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "intent_classifier.joblib"
_pipe = None

def predict(text: str):
    global _pipe
    text = (text or "").strip()
    if not text:
        return "-", "empty"
    cleaned = clean_text(text)
    if not cleaned:
        return "-", "empty after clean"
    if _pipe is None:
        _pipe = joblib.load(MODEL)
    label = str(_pipe.predict([cleaned])[0])
    if hasattr(_pipe, "predict_proba"):
        proba = _pipe.predict_proba([cleaned])[0]
        detail = " | ".join(f"{c}: {p:.3f}" for c, p in zip(_pipe.classes_, proba))
        return label, detail
    return label, ""

def main():
    with gr.Blocks(title="Intent Classification Engine") as demo:
        gr.Markdown("## Intent Classification Engine\nbenign vs injection")
        inp = gr.Textbox(lines=5, label="prompt")
        btn = gr.Button("classify")
        out1 = gr.Textbox(label="prediction")
        out2 = gr.Textbox(label="scores")
        btn.click(predict, inputs=inp, outputs=[out1, out2])
    demo.launch(server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()
