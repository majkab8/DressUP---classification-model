import io
import joblib
import requests
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from model_file import FashionModel

app = FastAPI()

model_path = "best_model.pth"
mlb_path = "mlb.pkl"

mlb = joblib.load(mlb_path)
fashion_model = FashionModel(model_path="tagging_model.pth", mlb=mlb, num_classes=len(mlb.classes_))


class AiBatchRequest(BaseModel):
    urls: List[str]


@app.post("/predict_batch")
async def predict_batch(req: AiBatchRequest):
    results = []

    for url in req.urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")

            labels_eng = fashion_model.predict(image)

            results.append({"url": url, "tags": labels_eng})

        except requests.exceptions.RequestException as e:
            results.append({"url": url, "tags": [], "error": f"Failed to download image: {e}"})
        except Exception as e:
            results.append({"url": url, "tags": [], "error": str(e)})

    return {"results": results}
