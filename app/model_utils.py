# model_utils.py
import torch
from PIL import Image
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


def predict_image(model, img_input, mlb, threshold=0.5, device="cpu"):
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        img = img_input

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        preds = (probs > threshold).astype(int)

    labels = [cls for cls, p in zip(mlb.classes_, preds) if p == 1]
    return labels


category_pl = {
    "tshirts": "koszulka",
    "shirt": "koszula",
    "jeans": "dżinsy",
    "hoodie": "bluza z kapturem",
    "sweater": "sweter",
    "skirt": "spódnica",
    "jacket": "kurtka",
    "coat": "płaszcz",
    "dress": "sukienka",
    "shorts": "szorty",
    "trousers": "spodnie",
    "leggings": "legginsy",
    "socks": "skarpetki",
    "bag": "torba",
    "handbag": "torebka",
    "wallet": "portfel",
    "watch": "zegarek",
    "belt": "pasek",
    "sunglasses": "okulary przeciwsłoneczne",
    "shoes": "buty",
    "sandals": "sandały",
    "flip flops": "klapki",
    "heels": "szpilki",
    "unknown_category": "nieznana kategoria"
}

color_pl = {
    "black": "czarny",
    "blue": "niebieski",
    "red": "czerwony",
    "green": "zielony",
    "yellow": "żółty",
    "grey": "szary",
    "brown": "brązowy",
    "white": "biały",
    "pink": "różowy",
    "purple": "fioletowy",
    "teal": "turkusowy",
    "orange": "pomarańczowy",
    "beige": "beżowy",
    "maroon": "bordowy",
    "navy": "granatowy",
    "cream": "kremowy",
    "unknown_color": "nieznany kolor"
}

style_pl = {
    "casual": "codzienny",
    "formal": "formalny",
    "sports": "sportowy",
    "party": "imprezowy",
    "ethnic": "etniczny",
    "street": "uliczny",
    "business": "biznesowy",
    "child": "dziecięcy",
    "unknown_style": "nieznany styl"
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
