# app/main.py
import io
from typing import Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from detector import YoloObjectDetector


app = FastAPI(
    title="YOLOv8 Object Detection API",
    version="1.0.0",
)

# Allow CORS for local Gradio UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create global detector instance (model loaded once)
detector = YoloObjectDetector(
    model_path="yolov8n.pt",  # COCO-pretrained nano model
    device="cpu",             # change to "cuda" if GPU available
    conf_threshold=0.25,
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receive an image, run YOLOv8 detection, return JSON detections.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    detections = detector.predict(image)
    return JSONResponse(content=detections)