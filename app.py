import io
import os
import subprocess
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import joblib

from classification import (
    predict_image,
    translate_labels
)

app = FastAPI(title="AI Tagging Model API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "best_model.pth"
mlb_path = "mlb.pkl"

if not (os.path.exists(model_path) and os.path.exists(mlb_path)):
    print("Model or label binarizer not found. Starting training process...")
    try:
        subprocess.run(["python", "train_model.py"], check=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
    except Exception as e:
        print(f"Unexpected error during training: {e}")

if os.path.exists(mlb_path):
    mlb = joblib.load(mlb_path)
    print(f"Loaded label binarizer with {len(mlb.classes_)} classes.")
else:
    mlb = None
    print("Warning: 'mlb.pkl' not found. Predictions may not work.")

num_classes = len(mlb.classes_) if mlb else 0
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Loaded trained model (best_model.pth).")
else:
    print("Warning: Model weights not found. Predictions will not work yet.")


@app.get("/")
def root():
    return {"message": "AI Tagging API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Ensure model and mlb exist
        if not os.path.exists(model_path) or mlb is None:
            return JSONResponse(
                {"error": "Model not trained yet. Please try again later."},
                status_code=503
            )

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run prediction
        labels = predict_image(model, image, mlb, device=device)
        translated = translate_labels(labels)
        return JSONResponse({
            "labels": labels,
            "labels_pl": translated
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
