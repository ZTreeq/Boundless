# YOLO + MobileNet Machine Learning Model

## Overview
- This repository contains a pipeline to train a YOLOv8 object detector and convert its labeled images into a classification dataset to train a MobileNet-based classifier.
- `main.py` implements: YOLO training, dataset conversion (detection -> classification), and MobileNet training.

## Repository layout
- `full-detection-final/`  — dataset (train / validation / test images and labels in YOLO format)
- `classes.txt`            — class names, one per line
- `main.py`                — training and conversion scripts
- `requirements.txt`       — Python dependencies

## Prerequisites
- Python 3.8+
- Recommended: GPU with CUDA for training
- Windows-specific notes:
  - To activate a virtual environment: `.\venv\Scripts\Activate.ps1` (PowerShell) or `.\venv\Scripts\activate` (cmd)

## Quick setup
1. Create & activate venv (PowerShell):
   - `python -m venv venv`
   - `.\venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`

## Dataset & paths
- `main.py` uses dataset paths pointing to `/content/...` (Colab). Update the top constants in `main.py` to the local paths on your machine, for example:
  - `YOLO_IMAGE_DIR = r"C:\Users\kattr\OneDrive\Desktop\YOLO + MobileNet Machine Learning Model\full-detection-final\train\images"`
  - `YOLO_LABEL_DIR = r"C:\Users\kattr\OneDrive\Desktop\YOLO + MobileNet Machine Learning Model\full-detection-final\train\labels"`
  - `CLASS_FILE = r"C:\Users\kattr\OneDrive\Desktop\YOLO + MobileNet Machine Learning Model\classes.txt"`
  - `CLASSIFIER_BASE = r"C:\Users\kattr\OneDrive\Desktop\YOLO + MobileNet Machine Learning Model\classifier"`
  - `DATA_YAML = r"C:\Users\kattr\OneDrive\Desktop\YOLO + MobileNet Machine Learning Model\data.yaml"`

## Example data.yaml (YOLO)
- A minimal `data.yaml` for YOLO training (save at the path assigned to `DATA_YAML`):
```yaml
train: path/to/train/images
val:   path/to/val/images
nc:  <number_of_classes>
names: [ "class0", "class1", ... ]
```

## How to run
- Edit paths at the top of `main.py` to match your workspace.
- From a Python REPL or a small runner file, call the functions:

PowerShell example:
- `python -c "from main import convert_yolo_to_classification; convert_yolo_to_classification()"`
- `python -c "from main import train_mobilenet; train_mobilenet()"`
- `python -c "from main import train_yolo; train_yolo()"`  (training YOLO may require CUDA and adjusted device index)

## Notes & outputs
- YOLO training will save weights (example: `best.pt`) in the working directory.
- MobileNet training saves `mobilenet_classifier.pth`.
- Adjust batch sizes / epochs / device IDs in `main.py` to match your hardware.

## Troubleshooting
- If ultralytics YOLO cannot see your dataset, double-check the `data.yaml` paths and that images/labels are accessible.
- For Windows path issues, prefer raw strings (prefix `r""`) or escape backslashes.

## License & attribution
- This repo uses models (YOLO, MobileNet) whose weights and code are provided by the respective open-source projects. Follow their licenses for redistribution and use.