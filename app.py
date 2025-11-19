import io

import joblib
import requests
from PIL import Image
from fastapi import FastAPI
from fastapi import Query
from fastapi.responses import JSONResponse

from fashion_classification import predict_single_image

app = FastAPI(title="AI Tagging Model API")

model_path = "best_model.pth"
mlb_path = "mlb.pkl"

mlb = joblib.load(mlb_path)
num_classes = len(mlb.classes_)

@app.get("/predict")
async def predict_raw(url: str = Query(..., description="Image URL to classify")):
    try:
        # Download image
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Run prediction
        labels = predict_single_image(image, model_path, mlb, num_classes)
        #translated = translate_labels(labels)

        return {"labels": labels}

    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Failed to download image: {e}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
