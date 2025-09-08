import torch
import torch.nn as nn
from torchvision import models, transforms
import yaml, os
from PIL import Image
import csv

from utils import GradCAM, find_duplicates


class Predictor:
    def __init__(self, config_path="params.yaml", labels_path="labels.yaml"):
        # ---------------------------
        # Load training config
        # ---------------------------
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["train"]

        # ---------------------------
        # Load labels mapping
        # ---------------------------
        self.labels = None
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                raw_map = yaml.safe_load(f).get("labels", {})

            # Force keys to int
            label_map = {int(k): str(v) for k, v in raw_map.items()}
            # Ensure order
            self.labels = [label_map[i] for i in sorted(label_map.keys())]

            print(f"[INFO] ✅ Loaded labels: {self.labels}")
        else:
            print("[WARN] ❌ labels.yaml not found, using generic IDs")
            self.labels = [f"class_{i}" for i in range(cfg["num_classes"])]

        # ---------------------------
        # Model setup
        # ---------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = cfg["model_path"]
        self.num_classes = cfg["num_classes"]
        self.image_size = cfg["image_size"]

        # Init backbone (ResNet18)
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()

        # Transforms
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

        # Forward
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            conf, pred = conf.item(), pred.item()

        # ✅ Human-readable label
        try:
            label = self.labels[pred]
        except IndexError:
            label = f"class_{pred}"
            print(f"[WARN] Prediction index {pred} out of label map range")

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
