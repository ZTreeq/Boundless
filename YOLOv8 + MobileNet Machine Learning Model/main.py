# main.py

# -----------------------------
# 🛠️ Setup and Dependencies
# -----------------------------
import os
import shutil
import random
import torch
from torch import nn, optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from ultralytics import YOLO

# -----------------------------
# 📁 Paths and Config
# -----------------------------
YOLO_IMAGE_DIR = "/content/dataset/full-detection-final/train/images"
YOLO_LABEL_DIR = "/content/dataset/full-detection-final/train/labels"
CLASS_FILE = "/content/classes.txt"
CLASSIFIER_BASE = "/content/dataset/classifier"
DATA_YAML = "/content/dataset/data.yaml"

# -----------------------------
# 📦 Step 1: Train YOLOv8 Model
# -----------------------------
def train_yolo():
    model = YOLO("yolov8n.pt")
    model.train(data=DATA_YAML, epochs=100, imgsz=640, batch=16, device=0)
    model.save("best.pt")  # Save trained weights

# -----------------------------
# 🧹 Step 2: Convert YOLO Dataset to Classification Format
# -----------------------------
def convert_yolo_to_classification():
    with open(CLASS_FILE, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    for split in ["train", "val"]:
        for class_name in class_names:
            os.makedirs(os.path.join(CLASSIFIER_BASE, split, class_name), exist_ok=True)

    image_class_map = {}

    for label_file in os.listdir(YOLO_LABEL_DIR):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(YOLO_LABEL_DIR, label_file)
        image_name = label_file.replace(".txt", ".jpg")
        image_path = os.path.join(YOLO_IMAGE_DIR, image_name)

        if not os.path.exists(image_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()
            class_ids = set(int(line.strip().split()[0]) for line in lines)

        for class_id in class_ids:
            class_name = class_names[class_id]
            image_class_map.setdefault(class_name, []).append(image_path)

    for class_name, image_paths in image_class_map.items():
        random.shuffle(image_paths)
        split_idx = int(0.8 * len(image_paths))
        train_images = image_paths[:split_idx]
        val_images = image_paths[split_idx:]

        for img_path in train_images:
            dest = os.path.join(CLASSIFIER_BASE, "train", class_name, os.path.basename(img_path))
            shutil.copy(img_path, dest)

        for img_path in val_images:
            dest = os.path.join(CLASSIFIER_BASE, "val", class_name, os.path.basename(img_path))
            shutil.copy(img_path, dest)

# -----------------------------
# 🧠 Step 3: Train MobileNet Classifier
# -----------------------------
def train_mobilenet():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = ImageFolder(os.path.join(CLASSIFIER_BASE, "train"), transform=transform)
    val_ds = ImageFolder(os.path.join(CLASSIFIER_BASE, "val"), transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, len(train_ds.classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10):
        model.train()
        loss_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        print(f"Epoch {epoch+1} Loss: {loss_total/len(train_loader):.4f}")

    torch.save(model.state_dict(), "mobilenet_classifier.pth")