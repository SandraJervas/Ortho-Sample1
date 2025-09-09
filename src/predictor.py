import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import csv
import yaml, os
import requests
from io import BytesIO
from src.utils import GradCAM, find_duplicates

class Predictor:
    def __init__(self, config_path="params.yaml"):
        # ---------------------------
        # Load training config
        # ---------------------------
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["train"]

        self.model_path = cfg["model_path"]
        self.num_classes = cfg["num_classes"]
        self.image_size = cfg.get("image_size", 224)

        # ---------------------------
        # Hardcoded class names
        # ---------------------------
        self.class_names = [
            "frontal_at_rest", "frontal_smile", "intraoral_front",
            "intraoral_left", "intraoral_right", "lower_jaw_view",
            "profile_at_rest", "upper_jaw_view"
        ][:self.num_classes]  # truncate if num_classes < 8
        print(f"[INFO] ✅ Using class names: {self.class_names}")

        # ---------------------------
        # Model setup
        # ---------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()

        # ---------------------------
        # Transforms
        # ---------------------------
        self.tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        os.makedirs("logs", exist_ok=True)

    def predict(self, image: Image.Image, threshold=0.7, gradcam=False):
        # ---------------------------
        # Preprocess
        # ---------------------------
        x = self.tf(image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            conf, pred = conf.item(), pred.item()

        # ---------------------------
        # Human-readable label
        # ---------------------------
        try:
            label = self.class_names[pred]
        except IndexError:
            label = f"class_{pred}"
            print(f"[WARN] Prediction index {pred} out of class_names range")

        # ---------------------------
        # Logs
        # ---------------------------
        logs = {
            "prediction_log": "logs/predictions.csv",
            "duplicates_log": "logs/duplicates.csv",
            "uncertain_log": "logs/uncertain.csv"
        }

        # Log prediction
        with open(logs["prediction_log"], "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label, conf])

        # Uncertain case
        if conf < threshold:
            with open(logs["uncertain_log"], "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([label, conf, "UNCERTAIN"])
            if gradcam:
                cam = GradCAM(self.model, target_layer="layer4")
                cam_path = cam.generate(x, pred, save_path="logs/gradcam.png")
                logs["gradcam"] = cam_path

        # Duplicate check
        dup_info = find_duplicates(image, label, conf, logs["duplicates_log"])
        if dup_info:
            logs["duplicate_removed"] = dup_info

        return {
            "label": label,
            "confidence": conf,
            "logs": logs
        }

    # ---------------------------
    # NEW: Predict multiple URLs
    # ---------------------------
    def predict_urls(self, urls, threshold=0.7, gradcam=False):
        """
        urls: list of dicts with 'url' key
        Returns predictions for each image URL
        """
        results = []

        for img_info in urls:
            url = img_info.get("url")
            if not url:
                continue
            try:
                response = requests.get(url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                results.append({"url": url, "status": "error", "error": str(e)})
                continue

            pred_res = self.predict(image, threshold=threshold, gradcam=gradcam)
            pred_res["url"] = url
            pred_res["status"] = "new" if pred_res["confidence"] >= threshold else "uncertain"
            results.append(pred_res)

        return {"images": results}
