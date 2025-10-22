# pip install torch torchvision torchaudio pytorch-lightning torchmetrics lightning pandas numpy scikit-learn matplotlib seaborn Pillow opencv-python kagglehub tqdm albumentations
import os
import re
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
import kagglehub
import matplotlib.pyplot as plt
import torch.multiprocessing

warnings.filterwarnings('ignore')

# POBRANIE DATASETU Z KAGGLEHUB
def load_dataset():
    path = kagglehub.dataset_download("nirmalsankalana/fashion-product-text-images-dataset")
    print("Path to dataset files:", path)

    # ZNALEZIENIE CSV
    csv_files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
    data_csv = os.path.join(path, csv_files[0])
    print("Found CSV file:", data_csv)

    df = pd.read_csv(data_csv)
    df = df.dropna(subset=["image", "description", "category"])
    df["image"] = df["image"].astype(str)

    return df, path

# KATEGORIE
COLORS = [
    "black", "blue", "red", "green", "yellow", "grey", "brown", "white",
    "pink", "purple", "teal", "orange", "beige", "maroon", "navy", "cream"
]

STYLES = [
    "casual", "formal", "sports", "party", "ethnic", "street", "business", "child"
]

EXTRA_CATEGORIES = [
    "t-shirt", "shirt", "jeans", "hoodie", "sweater", "skirt", "jacket", "coat",
    "dress", "shorts", "trousers", "leggings", "socks", "bag", "handbag", "wallet",
    "watch", "belt", "sunglasses", "shoes", "sandals", "flip flops", "heels"
]

# KATEGORIE NA STYLE MAPOWANIE
CATEGORY_TO_STYLES = {
    "shirt": ["business", "formal"],
    "t-shirt": ["casual", "sports"],
    "jeans": ["casual", "street"],
    "trousers": ["casual", "business", "formal"],
    "hoodie": ["casual", "sports", "street"],
    "sweater": ["casual", "business"],
    "skirt": ["casual", "formal"],
    "dress": ["formal", "party"],
    "jacket": ["casual", "business", "street"],
    "coat": ["formal", "business"],
    "shorts": ["casual", "sports"],
    "shoes": ["formal", "casual", "sports"],
    "sandals": ["casual", "sports"],
    "flip flops": ["casual"],
    "heels": ["formal", "party"]
}

# SZUKANIE SŁOW KLUCZOWYCH W TEKŚCIE
def extract_keywords(text, keywords, default):
    text = str(text).lower()
    return [kw for kw in keywords if kw in text] or [default]

# ZROBIENIE LABELI
def build_labels(row):

    desc = str(row["description"]).lower()
    base = [row["category"].lower()]
    colors = extract_keywords(desc, COLORS, "unknown_color")
    styles = extract_keywords(desc, STYLES, "unknown_style")

    # JEDEN TYP UBRANIA
    extra = [next((cat for cat in EXTRA_CATEGORIES if cat in desc), "unknown_category")]

    # WYBRANIE STYLI PATRZĄC NA KATEGORIE
    auto_styles = sum((CATEGORY_TO_STYLES.get(cat, []) for cat in base + extra), [])

    return list(set(base + colors + styles + extra + auto_styles))

# TRANSFORMACJE DANYCH
transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# DATASET
class FashionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, mlb=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        self.files = self.df["image"].astype(str).tolist()
        self.labels_list = self.df["labels"].tolist()

        # MultiLabelBinarizer
        if mlb is None:
            self.mlb = MultiLabelBinarizer()
            self.labels_binary = self.mlb.fit_transform(self.labels_list)
        else:
            self.mlb = mlb
            self.labels_binary = self.mlb.transform(self.labels_list)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        img_path = os.path.join(self.img_dir, file_name)

        # WCZYTANIE OBRAZU
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        labels = torch.FloatTensor(self.labels_binary[idx])
        return image, labels


# TRENING JEDNEJ EPOKI
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# WALIDACJA
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > threshold).int().cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    report = classification_report(all_labels, all_preds, target_names=train_dataset.mlb.classes_, zero_division=0)
    return total_loss / len(loader), acc, f1, report

# TRENING
def main_training():

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(train_dataset.mlb.classes_)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Zamrożenie warstw feature extractor
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

    EPOCHS = 5
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n Epoch {epoch}/{EPOCHS}")

        # TRENING
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")

        # WALIDACJA
        val_loss, val_acc, val_f1, report = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc:  {val_acc:.4f} | F1: {val_f1:.4f}")
        print(report)

        # ZAPIS NAJLEPSZEGO MODELU
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Zapisano najlepszy model.")

    print("\n Trening zakończony.")
    return model

def predict_image(model, img_path, mlb, threshold=0.5, device="cpu"):

    img = Image.open(img_path).convert("RGB")
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

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, path = load_dataset()

    df["labels"] = df.apply(build_labels, axis=1)

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    img_dir = os.path.join(path, "data")

    mlb = MultiLabelBinarizer()
    mlb.fit(train_df["labels"].tolist())

    train_dataset = FashionDataset(train_df, img_dir, transform=transform, mlb=mlb)
    val_dataset = FashionDataset(val_df, img_dir, transform=transform, mlb=mlb)
    test_dataset = FashionDataset(test_df, img_dir, transform=transform, mlb=mlb)

    model = main_training()
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    img_path = "C:/Users/Maja/Desktop/image1.jpg"
    labels = predict_image(model, img_path, mlb, device=device)
    print("Predykcje:", labels)