# api.py
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2, base64
import numpy as np
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load YOLO
yolo = YOLO("best.pt")

# Load MobileNet
mobilenet = models.mobilenet_v2(pretrained=False)
mobilenet.classifier[1] = torch.nn.Linear(mobilenet.last_channel, 8)  # adjust to your num classes
mobilenet.load_state_dict(torch.load("mobilenet_classifier.pth", map_location="cpu"))
mobilenet.eval()

# Labels
debris_labels = ["silica_glass","pharma_crystal","semiconductor","aerospace_metal","large_structural","composite_material","satellite","rocket"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = yolo.predict(img)
    output = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        crop = img[y1:y2, x1:x2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = transform(crop_pil).unsqueeze(0)
        outputs = mobilenet(tensor)
        _, pred = torch.max(outputs, 1)
        debris_type = debris_labels[pred.item()]
        output.append({"box": [x1, y1, x2, y2], "label": debris_type})

        # Draw bounding box and label on the image
        color = (80, 51, 255)  # #5033FF in BGR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_text = debris_type
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label_text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    # Encode annotated image to base64
    _, buffer = cv2.imencode('.png', img)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "annotated_image": annotated_b64,
        "detections": output
    })

@app.route("/")
def index():
    return """
    <html>
        <body style="background-color: white;">
        </body>
    </html>
    """

if __name__ == "__main__":
    app.run(port=8080,use_reloader=False) 
