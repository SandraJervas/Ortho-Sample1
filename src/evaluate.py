import os, json, yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.nn as nn

class Evaluator:
    def __init__(self, config_path="params.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["evaluate"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_dir = cfg["test_dir"]
        self.batch_size = cfg["batch_size"]
        self.img_size = cfg["image_size"]
        self.model_path = cfg["model_path"]
        self.metrics_output = cfg["metrics_output"]

        os.makedirs(os.path.dirname(self.metrics_output), exist_ok=True)
        self._init_data()
        self._init_model()

    def _init_data(self):
        norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        test_tf = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            norm
        ])
        self.test_loader = DataLoader(
            datasets.ImageFolder(self.test_dir, transform=test_tf),
            batch_size=self.batch_size,
            shuffle=False
        )

    def _init_model(self):
        self.model = models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.test_loader.dataset.classes))
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    def evaluate(self):
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in self.test_loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                _, preds = torch.max(self.model(imgs), 1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        acc = correct / total
        json.dump({"test_accuracy": acc}, open(self.metrics_output, "w"), indent=4)
        print(f"Test Accuracy: {acc:.4f}")
        return acc

if __name__ == "__main__":
    Evaluator().evaluate()
