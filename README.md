# Space Debris Detector

Lightweight Chrome extension + Flask API that lets users upload an image, runs object detection + classifier on the server, and returns an annotated image (bounding boxes + labels). UI uses custom fonts included in `chrome_extension/` and a purple (#5033FF) accent palette.

## Features
- Upload image from Chrome extension popup
- Server runs YOLO detection and MobileNet classifier on crops
- Returns annotated image (base64 PNG) and detection metadata
- CORS enabled for local extension <> API communication
- Fonts included: `ThuastDemo-jE5zy.otf` (title) and `NotoSansMono.ttf` (body)

## Repository structure
- app.py — Flask API (serves /predict)
- requirements.txt — Python dependencies (add contents as needed)
- best.pt — YOLO weights (NOT included; add your trained model)
- mobilenet_classifier.pth — MobileNet classifier weights (NOT included)
- chrome_extension/
  - popup.html
  - popup.js
  - style.css
  - ThuastDemo-jE5zy.otf
  - NotoSansMono.ttf
  - manifest.json

## API
POST /predict
- Form field: `file` — image file to analyze
- Response (JSON):
  - `annotated_image`: base64 PNG string with boxes & labels
  - `detections`: array of { box: [x1,y1,x2,y2], label: "<class>" }

Example curl (test):
```bash
curl -X POST -F "file=@/path/to/image.png" http://127.0.0.1:5000/predict
```

## Setup (Windows)
1. Clone repo
2. Create virtual environment and activate:
```powershell
python -m venv .venv
.venv\Scripts\activate
```
3. Install dependencies (examples — adjust for your environment and GPU support):
```powershell
pip install flask flask-cors pillow opencv-python torchvision torch ultralytics
```
(Alternatively add these into `requirements.txt` and run `pip install -r requirements.txt`.)

4. Place model files next to `app.py`:
- `best.pt` (YOLO)
- `mobilenet_classifier.pth` (MobileNet classifier)

5. Run the API:
```powershell
python app.py
```
The API listens on http://0.0.0.0:5000 — open http://127.0.0.1:5000/ to check status.

## Install Chrome extension (developer)
1. Open Chrome → Extensions → Load unpacked.
2. Select the `chrome_extension/` folder.
3. Open the extension popup, choose an image, and click Upload. The popup displays the original and the annotated image.

Note: extension popup dimensions are intentionally wide for clearer images. If Chrome restricts the popup size on some systems, use the web API (curl or a simple web page) or adjust CSS.

## Troubleshooting
- "No annotated image returned": Check Flask logs for exceptions and ensure model files exist and load correctly.
- CORS errors: The Flask app enables CORS for all origins; ensure requests target the correct host/port.
- Large models / GPU: Installing `torch` with CUDA requires matching CUDA toolkit versions — consult PyTorch install docs.

## Security & Privacy
- This repo is a local demo. Do not expose the API publicly without authentication.
- Uploaded images are processed locally by the server and not stored by default (unless you add persistence).

## Contributing
Fixes and improvements welcome. Open an issue or submit a PR.

## References
This project benefitted from AI-assisted development using ChatGPT. Guidance included:

- Debugging Python errors and environment setup
- Writing scripts to convert detection datasets into classification folders
- Handling file uploads

All code was reviewed and adapted by the author to meet the specific goals of the NASA Space Apps Hackathon.
