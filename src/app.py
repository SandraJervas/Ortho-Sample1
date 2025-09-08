from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests, io, csv, os, hashlib
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
import yaml

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
# Predictor class
# -----------------------------
class Predictor:
    def __init__(self, config_path="params.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["train"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = cfg["model_path"]
        self.num_classes = cfg["num_classes"]
        self.image_size = cfg["image_size"]

        # TODO: Replace with real class labels (or load from labels.yaml)
        self.labels = [f"class_{i}" for i in range(self.num_classes)]  

        # init model
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device).eval()

        # transforms
        self.tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        os.makedirs("logs", exist_ok=True)

    def predict(self, image: Image.Image):
        x = self.tf(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
        return self.labels[pred.item()], conf.item()

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

#  GET Welcome Endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Orthodontic Classifier API 🚀"}

#  POST Prediction Endpoint
@app.post("/predict")
async def predict_batch(req: PredictRequest):
    results = []
    seen = {}  # hash -> (url, label, conf)

    for item in req.images:
        try:
            # download
            resp = requests.get(item.url, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")

            # predict
            label, conf = predictor.predict(image)

            # log prediction
            with open("logs/predictions.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([item.url, label, conf])

            # hash for duplicate detection
            h = get_hash(image)

            # check duplicates
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

            # first time seeing this image
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
