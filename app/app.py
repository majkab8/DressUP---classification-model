# app.py
import io
import joblib
import torch
from PIL import Image
import requests
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from model_utils import predict_single_image, transform

category_pl = {
    "tshirts": "koszulka", "shirt": "koszula", "jeans": "dżinsy",
    "hoodie": "bluza z kapturem", "sweater": "sweter", "skirt": "spódnica",
    "jacket": "kurtka", "coat": "płaszcz", "dress": "sukienka",
    "shorts": "szorty", "trousers": "spodnie", "leggings": "legginsy",
    "socks": "skarpetki", "bag": "torba", "handbag": "torebka",
    "wallet": "portfel", "watch": "zegarek", "belt": "pasek",
    "sunglasses": "okulary przeciwsłoneczne", "shoes": "buty",
    "sandals": "sandały", "flip flops": "klapki", "heels": "szpilki",
    "unknown_category": "nieznana kategoria"
}

color_pl = {
    "black": "czarny", "blue": "niebieski", "red": "czerwony",
    "green": "zielony", "yellow": "żółty", "grey": "szary",
    "brown": "brązowy", "white": "biały", "pink": "różowy",
    "purple": "fioletowy", "teal": "turkusowy", "orange": "pomarańczowy",
    "beige": "beżowy", "maroon": "bordowy", "navy": "granatowy",
    "cream": "kremowy", "unknown_color": "nieznany kolor"
}

style_pl = {
    "casual": "codzienny", "formal": "formalny", "sports": "sportowy",
    "party": "imprezowy", "ethnic": "etniczny", "street": "uliczny",
    "business": "biznesowy", "child": "dziecięcy", "unknown_style": "nieznany styl"
}

extra_pl = {**category_pl}


def translate_labels(labels):
    translated = []
    for lab in labels:
        if lab in category_pl:
            translated.append(category_pl[lab])
        elif lab in color_pl:
            translated.append(color_pl[lab])
        elif lab in style_pl:
            translated.append(style_pl[lab])
        elif lab in extra_pl:
            translated.append(extra_pl[lab])
        else:
            translated.append(lab)
    return translated


app = FastAPI(title="AI Tagging Model API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlb_path = "mlb.pkl"
mlb = joblib.load(mlb_path)
num_classes = len(mlb.classes_)

model_path = "tagging-model.pth"


@app.get("/predict")
async def predict_raw(url: str = Query(..., description="Image URL to classify")):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        labels = predict_single_image(image, model_path, mlb, num_classes)
        labels_pl = translate_labels(labels)

        return {"labels": labels, "labels_pl": labels_pl}

    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Failed to download image: {e}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
