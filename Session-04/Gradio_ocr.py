from __future__ import annotations

import base64
import io
import os

import gradio as gr
from PIL import Image
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_OCR_MODEL = "gpt-4o-mini"

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def run_ocr(image: Image.Image | None) -> str:
    if image is None:
        return "Please upload an image."
    if openai_client is None:
        return "OCR failed: OPENAI_API_KEY is not set in your .env file."

    try:
        rgb_image = image.convert("RGB")
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = openai_client.responses.create(
            model=OPENAI_OCR_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Perform OCR on this image and you are a number plate extractor. so only extract number plate from images. The images provided are from pakistan cities. Return only the extracted text, "
                                "preserving line breaks when possible. Do not add explanations."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )
        text = (response.output_text or "").strip()
        if not text:
            return "No text detected."
        return text
    except Exception as error:
        return f"OCR failed: {error}"


with gr.Blocks() as demo:
    gr.Markdown("# OCR with Gradio + OCR Vision")
    gr.Markdown(
        "Upload an image and extract text using Vision OCR ")

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
