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
encoders_path = "encoders.pkl"

encoders = joblib.load(encoders_path)
fashion_model = FashionModel(model_path=model_path, encoders=encoders)

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

            labels_dict = fashion_model.predict(image)

            results.append({
                "url": url,
                "tags": labels_dict
            })

        except requests.exceptions.RequestException as e:
            results.append({
                "url": url,
                "tags": {},
                "error": f"Failed to download image: {e}"
            })
        except Exception as e:
            results.append({
                "url": url,
                "tags": {},
                "error": str(e)
            })

    return {"results": results}
