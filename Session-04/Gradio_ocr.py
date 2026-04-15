from __future__ import annotations

import gradio as gr
from PIL import Image
from transformers import pipeline

MODEL_ID = "microsoft/trocr-base-printed"
ocr_pipeline = None
TASK_CANDIDATES = ("image-text-to-text", "image-to-text")

from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def get_ocr_pipeline():
    """Load OCR pipeline once and reuse it."""
    global ocr_pipeline
    if ocr_pipeline is None:
        last_error = None
        for task_name in TASK_CANDIDATES:
            try:
                ocr_pipeline = pipeline(task_name, model=MODEL_ID)
                break
            except Exception as error:
                last_error = error
        if ocr_pipeline is None:
            raise RuntimeError(
                "Could not create OCR pipeline with supported task names "
                f"{TASK_CANDIDATES}. Last error: {last_error}"
            )
    return ocr_pipeline


def run_ocr(image: Image.Image | None) -> str:
    if image is None:
        return "Please upload an image."

    try:
        predictor = get_ocr_pipeline()
        result = predictor(image)
        if not result:
            return "No text detected."
        first_result = result[0]
        if isinstance(first_result, dict):
            return first_result.get("generated_text") or first_result.get("text") or "No text detected."
        return str(first_result)
    except Exception as error:
        return f"OCR failed: {error}"


with gr.Blocks() as demo:
    gr.Markdown("# OCR with Gradio + OCR model")
    gr.Markdown(
        "Upload an image and extract text using the OCR model "
        f"`{MODEL_ID}`."
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Input Image")
        text_output = gr.Textbox(
            label="OCR Result",
            lines=12,
            placeholder="Recognized text will appear here...",
        )

    run_button = gr.Button("Run OCR")
    run_button.click(fn=run_ocr, inputs=image_input, outputs=text_output)


if __name__ == "__main__":
    demo.launch()
