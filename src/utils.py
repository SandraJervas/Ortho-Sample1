# src/utils.py
import hashlib
import numpy as np
from PIL import Image

# Dummy GradCAM class (placeholder, extend later if needed)
class GradCAM:
    def __init__(self, model, target_layer="layer4"):
        self.model = model
        self.target_layer = target_layer

    def generate(self, x, pred, save_path="logs/gradcam.png"):
        # TODO: implement real GradCAM
        with open(save_path, "w") as f:
            f.write("GradCAM placeholder")
        return save_path


def find_duplicates(image, label, conf, log_path="logs/duplicates.csv"):
    """Basic hash-based duplicate check (extend later)."""
    img = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.array(img).flatten()
    avg = pixels.mean()
    bits = "".join(['1' if p > avg else '0' for p in pixels])
    h = hashlib.sha256(bits.encode()).hexdigest()

    # For now, just return None so nothing breaks
    return None
