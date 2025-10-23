import io

import joblib
import requests
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from fastapi import FastAPI
from fastapi import Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from classification import predict_image, translate_labels

app = FastAPI(title="AI Tagging Model API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model.pth"
mlb_path = "mlb.pkl"

mlb = joblib.load(mlb_path)
num_classes = len(mlb.classes_)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

class PredictRequest(BaseModel):
    url: str

@app.get("/predict")
async def predict_raw(url: str = Query(..., description="Image URL to classify")):
    try:
        # Download image
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Run prediction
        labels = predict_image(model, image, mlb, device=device)
        translated = translate_labels(labels)

        return {"labels": labels, "labels_pl": translated}

    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Failed to download image: {e}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
