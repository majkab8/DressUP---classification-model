import os
from tqdm import tqdm
import kagglehub
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# POBRANIE DATASETU Z KAGGLEHUB
def load_dataset():
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
    csv_file = next(f for f in os.listdir(path) if f.endswith(".csv"))
    df = pd.read_csv(os.path.join(path, csv_file)).dropna(subset=["image", "description", "category"])
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
    return [k for k in keywords if k in str(text).lower()] or [default]


# ZROBIENIE LABELI
def build_labels(row):
    desc = str(row["description"]).lower()
    base = [row["category"].lower()]
    colors = extract_keywords(desc, COLORS, "unknown_color")
    styles = extract_keywords(desc, STYLES, "unknown_style")
    # JEDEN TYP UBRANIA
    extra = [next((c for c in EXTRA_CATEGORIES if c in desc), "unknown_category")]
    # WYBRANIE STYLI PATRZĄC NA KATEGORIE
    auto_styles = sum((CATEGORY_TO_STYLES.get(c, []) for c in base + extra), [])
    return list(set(base + colors + styles + extra + auto_styles))


# TRANSFORMACJE DANYCH
train_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DATASET
class FashionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, mlb=None):
        self.imgs = df["image"].astype(str).tolist()
        self.labels = df["labels"].tolist()
        self.transform = transform
        self.img_dir = img_dir
        self.mlb = mlb or MultiLabelBinarizer().fit(self.labels)
        self.bin_labels = self.mlb.transform(self.labels)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        # WCZYTANIE OBRAZU
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        img = self.transform(img)
        return img, torch.FloatTensor(self.bin_labels[idx])


# TRENING JEDNEJ EPOKI
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# EWALUACJA
def evaluate(model, loader, criterion, device, mlb, threshold=0.3):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            total_loss += criterion(outputs, labels).item()
            all_preds.append((torch.sigmoid(outputs) > threshold).cpu())
            all_labels.append(labels.cpu())
    labels_true, labels_pred = torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()

    return {
        "loss": total_loss / len(loader),
        "acc": accuracy_score(labels_true, labels_pred),
        "f1": f1_score(labels_true, labels_pred, average="macro"),
        "report": classification_report(labels_true, labels_pred, target_names=mlb.classes_, zero_division=0)
    }

# TRENING
def main_training(train_dataset, val_dataset, test_dataset, device):
    loaders = {
        "train": DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True),
        "test": DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    }

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.mlb.classes_))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": []
    }
    best_val_loss = float("inf")

    for epoch in range(1, 21):
        print(f"\nEpoch {epoch}")
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        train_metrics = evaluate(model, loaders["train"], criterion, device, train_dataset.mlb)
        val_metrics = evaluate(model, loaders["val"], criterion, device, train_dataset.mlb)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_metrics["acc"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1"].append(val_metrics["f1"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), "best_model.pth")
            print("Zapisano najlepszy model.")

    print("\nTrening zakończony.")

    plot_curves(history)
    test_metrics = evaluate(model, loaders["test"], criterion, device, train_dataset.mlb)
    print(f"Test Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['acc']:.4f} | F1: {test_metrics['f1']:.4f}")
    print(test_metrics["report"])
    return model


def plot_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    # Wykres strat
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Wykres metryk
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.plot(epochs, history["train_f1"], label="Train F1")
    plt.plot(epochs, history["val_f1"], label="Val F1")
    plt.title("Accuracy and F1 per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig("training_curves.png")

def predict_image(model, img, mlb, threshold=0.3, device="cpu"):
    img = Image.open(img).convert("RGB") if isinstance(img, str) else img
    x = train_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds_raw = torch.sigmoid(model(x)).cpu().numpy()[0]
        preds = preds_raw > threshold

    selected = [cls for cls, p in zip(mlb.classes_, preds) if p]

    if not selected:
        best_idx = preds_raw.argmax()
        selected = [mlb.classes_[best_idx]]

    return selected

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
    for label in labels:
        if label in category_pl:
            translated.append(category_pl[label])
        elif label in color_pl:
            translated.append(color_pl[label])
        elif label in style_pl:
            translated.append(style_pl[label])
        elif label in extra_pl:
            translated.append(extra_pl[label])
        else:
            translated.append(label)
    return translated
