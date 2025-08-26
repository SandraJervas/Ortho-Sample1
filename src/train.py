import os, json, yaml, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
from tqdm import tqdm

# --- Ensure all images are RGB ---
class ToRGB:
    def __call__(self, img):
        return img.convert("RGB")

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience, self.counter, self.best_acc, self.early_stop = patience, 0, 0, False
    def __call__(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc, self.counter = val_acc, 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience

# --- Trainer ---
class Trainer:
    def __init__(self, config_path="params.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["train"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dir, self.val_dir = cfg["train_dir"], cfg["val_dir"]
        self.batch_size, self.lr = cfg["batch_size"], cfg["lr"]
        self.epochs, self.num_classes = cfg["num_epochs"], cfg["num_classes"]
        self.img_size, self.model_path = cfg["image_size"], cfg["model_path"]
        self.metrics_output, self.patience = cfg["metrics_output"], cfg.get("patience", 5)
        self.resume, self.backbone = cfg.get("resume", False), cfg.get("backbone", "resnet18")

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_output), exist_ok=True)

        self._init_data()
        self._init_model()

    def _init_data(self):
        norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        train_tf = transforms.Compose([
            ToRGB(),
            transforms.Resize((self.img_size,self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(),
            norm
        ])
        val_tf = transforms.Compose([
            ToRGB(),
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            norm
        ])
        self.train_loader = DataLoader(datasets.ImageFolder(self.train_dir, transform=train_tf),
                                       batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(datasets.ImageFolder(self.val_dir, transform=val_tf),
                                     batch_size=self.batch_size)

    def _init_model(self):
        backbone_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50}
        if self.backbone not in backbone_dict:
            raise ValueError(f"Unsupported backbone {self.backbone}")

        self.model = backbone_dict[self.backbone](weights="IMAGENET1K_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.resume and os.path.exists(self.model_path):
            print(f" Resuming from {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def _save_model(self, val_acc):
        ts = time.strftime("%Y%m%d_%H%M%S")
        unique = self.model_path.replace(".pth", f"_{ts}.pth")
        torch.save(self.model.state_dict(), unique)
        torch.save(self.model.state_dict(), self.model_path)  # overwrite latest
        print(f" Model improved ({val_acc:.4f}), saved {unique} + updated {self.model_path}")

    def train(self):
        stopper, best_acc = EarlyStopping(self.patience), 0.0
        for epoch in range(1, self.epochs+1):
            self.model.train(); run_loss = 0
            for imgs, lbls in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}"):
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(imgs), lbls)
                loss.backward(); self.optimizer.step()
                run_loss += loss.item()
            val_acc = self.evaluate()
            print(f"Epoch {epoch}: Loss {run_loss:.3f}, ValAcc {val_acc:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc; self._save_model(val_acc)
            stopper(val_acc)
            if stopper.early_stop: 
                print("Early stopping."); break

        json.dump({"val_accuracy": best_acc}, open(self.metrics_output,"w"), indent=4)

    def evaluate(self):
        self.model.eval(); correct=total=0
        with torch.no_grad():
            for imgs, lbls in self.val_loader:
                imgs,lbls=imgs.to(self.device),lbls.to(self.device)
                _, pred = torch.max(self.model(imgs), 1)
                correct += (pred==lbls).sum().item(); total += lbls.size(0)
        return correct/total

if __name__=="__main__":
    Trainer().train()
