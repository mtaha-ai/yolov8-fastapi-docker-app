# ui/app.py
import io
import json
import uuid
from pathlib import Path
from typing import Dict, Tuple, List

import requests
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

API_URL = "http://localhost:8000/predict"
OUTPUT_DIR = Path("ui_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_boxes(
    image: Image.Image,
    detections: Dict,
) -> Image.Image:
    """
    Draw bounding boxes and labels on the image using the detections dict
    returned by the API.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections.get("detections", []):
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        label = det["class_name"]
        conf = det["confidence"]

        text = f"{label} {conf:.2f}"

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Text background
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        text_bg = [x1, y1 - th, x1 + tw, y1]
        draw.rectangle(text_bg, fill="red")
        # Text
        draw.text((x1, y1 - th), text, fill="white", font=font)

    return annotated


def save_detections_json(detections: Dict) -> str:
    """
    Save detections dictionary to a JSON file and return its path as string.
    """
    file_id = uuid.uuid4().hex
    json_path = OUTPUT_DIR / f"detections_{file_id}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    return str(json_path)


def predict_gradio(image: Image.Image):
    """
    Gradio callback:
      - sends image to FastAPI backend
      - gets detections JSON
      - draws bounding boxes
      - saves detections JSON for download
    """
    if image is None:
        return None, None

    # Encode image to send via HTTP
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("image.png", buf, "image/png")}
    resp = requests.post(API_URL, files=files)
    resp.raise_for_status()
    detections = resp.json()

    annotated = draw_boxes(image, detections)
    json_path = save_detections_json(detections)

    return annotated, json_path


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# YOLOv8 Object Detection (Gradio UI + FastAPI backend)")
        gr.Markdown(
            "Upload an image, run detection via FastAPI backend, "
            "see bounding boxes and download detections as JSON."
        )

        with gr.Row():
            with gr.Column():
                img_input = gr.Image(
                    label="Input Image",
                    type="pil",
                )
                btn = gr.Button("Detect")

            with gr.Column():
                img_output = gr.Image(
                    label="Image with Detections",
                    type="pil",
                )
                json_file = gr.File(
                    label="Download detections (JSON)"
                )

        btn.click(
            fn=predict_gradio,
            inputs=img_input,
            outputs=[img_output, json_file],
        )

    demo.launch(server_port=7860, share=False)


if __name__ == "__main__":
    main()