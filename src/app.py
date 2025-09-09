from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests, io, csv, os, hashlib
import numpy as np
from PIL import Image
import yaml
from src.predictor import Predictor


# -----------------------------
# Helper: compute image hash
# -----------------------------
def get_hash(image: Image.Image) -> str:
    """Compute hash for duplicate detection (average hash)."""
    img = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.array(img).flatten()
    avg = pixels.mean()
    bits = "".join(['1' if p > avg else '0' for p in pixels])
    return hashlib.sha256(bits.encode()).hexdigest()


# -----------------------------
# API Input Schema
# -----------------------------
class ImageItem(BaseModel):
    url: str

class PredictRequest(BaseModel):
    images: List[ImageItem]


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Orthodontic Classifier API")
predictor = Predictor("params.yaml")


@app.get("/")
async def root():
    return {"message": "Welcome to Orthodontic Classifier API 🚀"}

@app.post("/predict")
async def predict_batch(req: PredictRequest):
    results = []
    seen = {}

    for item in req.images:
        try:
            resp = requests.get(item.url, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")

            # compute hash
            h = get_hash(image)

            # duplicate check
            if h in seen:
                results.append({
                    "url": item.url,
                    "label": seen[h]["label"],
                    "confidence": seen[h]["confidence"],
                    "status": "duplicate"
                })
                continue

            # prediction
            pred = predictor.predict(image)
            label, conf = pred["label"], pred["confidence"]

            # log prediction
            with open("logs/predictions.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([item.url, label, conf])

            seen[h] = {"label": label, "confidence": conf}

            results.append({
                "url": item.url,
                "label": label,
                "confidence": conf,
                "status": "new" if conf >= 0.7 else "uncertain"
            })

        except Exception as e:
            results.append({"url": item.url, "status": "error", "error": str(e)})

    return {"images": results}
