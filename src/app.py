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
predictor = Predictor("params.yaml", "labels.yaml")

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

            # ✅ Now this uses predictor from predictor.py
            pred = predictor.predict(image)
            label, conf = pred["label"], pred["confidence"]

            # log prediction
            with open("logs/predictions.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([item.url, label, conf])

            # duplicate check
            h = get_hash(image)
            if h in seen:
                prev_url, prev_label, prev_conf = seen[h]
                if conf > prev_conf:
                    with open("logs/duplicates.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([prev_url, prev_label, prev_conf, "removed"])
                        writer.writerow([item.url, label, conf, "kept"])
                    seen[h] = (item.url, label, conf)
                else:
                    with open("logs/duplicates.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([item.url, label, conf, "removed"])
                continue

            seen[h] = (item.url, label, conf)

        except Exception as e:
            results.append({"url": item.url, "error": str(e)})

    # prepare unique output
    for url, label, conf in seen.values():
        results.append({
            "url": url,
            "label": label,
            "confidence": conf,
            "status": "new"
        })

    return {"images": results}
