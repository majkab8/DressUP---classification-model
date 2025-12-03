import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchmetrics import Accuracy, F1Score
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
import pytorch_lightning as pl
from torchvision.models import ResNet18_Weights

register_heif_opener()

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FashionDataset(Dataset):
  def __init__(self, *, df, img_dir, transforms, is_test=False):
    self.df = df
    self.img_dir = img_dir
    self.transform = transforms
    self.is_test = is_test

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    img_id = str(row['id'])
    img_path = None

    for ext in ['.jpg', '.jpeg', '.png', '.heic', '.HEIC']:
      temp_path = os.path.join(self.img_dir, f"{img_id}{ext}")
      if os.path.exists(temp_path):
        img_path = temp_path
        break

    if img_path is None:
      img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

    img = Image.open(img_path).convert("RGB")
    img = self.transform(img)

    label_vector = torch.tensor(row.get("label_vector", [0] * self.df.shape[1]), dtype=torch.float32)

    if self.is_test:
      return img, label_vector, img_path
    else:
      return img, label_vector

class FashionDataModule(pl.LightningDataModule):
  def __init__(self, *, num_classes, batch_size=32, num_workers=0, pin_memory=False, train_transform, val_test_transform, images_path, labels_path):
    super().__init__()

    self.num_classes = num_classes
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.train_transform = train_transform
    self.val_test_transform = val_test_transform
    self.labels_path = labels_path
    self.images_path = images_path

  def prepare_data(self):
    pass

  def setup(self, stage=None):
    np.random.seed(42)
    df = pd.read_csv(self.labels_path, on_bad_lines='skip')
    df = df.dropna(subset=['id', 'articleType'])
    exclude_classes = ["Watches", "Socks", "Shoe Accessories", "Deodorant", "Lipstick", "Briefs", "Perfume and Body Mist", "Nail Polish", "Laptop Bag", "Wallets", "Ring", "Clutches", "Earrings", "Boxers", "Jewellery Set", "Duppatta", "Lip Gloss", "Bath Robe", "Face Wash and Cleanser", "Necklace and Chains", "Foundation and Primer", "Free Gifts", "Trolley Bag", "Shoe Laces", "Fragrance Gift Set", "Baby Dolls", "Highligher and Blush", "Travel Accessory", "Mobile Pouch", "Lip Care", "Beauty Accessory", "Kajal and Eyeliner", "Water Bottle", "Lip Liner", "Stockings", "Eyeshadow", "Nail Essentials", "Face Scrub and Exfoliator", "Mask and Peel", "Wristbands", "Tablet Sleeve", "Footballs", "Hair Colour", "Concealer", "Body Lotion", "Sunscreen", "Hair Accessory", "Basketballs"]
    df = df[~df['articleType'].isin(exclude_classes)].reset_index(drop=True)

    for col in ["season", "baseColour", "subCategory", "articleType", "usage"]:
            if col not in df.columns:
                df[col] = "unknown"

    def build_labels(row):
        labels = [
          str(row["season"]).lower(),
          str(row["baseColour"]).lower(),
          str(row["subCategory"]).lower(),
          str(row["articleType"]).lower(),
          str(row["usage"]).lower()
        ]
        return list(set(labels))

    df["labels"] = df.apply(build_labels, axis=1)

    df = df.head(10000)

    def check_file_exists(row):
      img_id = str(row['id'])
      for ext in ['.jpg', '.jpeg', '.png', '.heic', '.HEIC']:
        full_path = os.path.join(self.images_path, img_id + ext)
        if os.path.exists(full_path):
          return True
      return False

    df['exists'] = df.apply(check_file_exists, axis=1)
    df = df[df['exists']].reset_index(drop=True)

    print("Pozostało obrazów:", len(df))

    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(df["labels"])
    df["label_vector"] = list(label_matrix)

    self.mlb = mlb
    self.num_classes = len(mlb.classes_)

    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))

    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size + val_size].reset_index(drop=True)
    test_df = df[train_size + val_size:].reset_index(drop=True)

    self.train_dataset = FashionDataset(df=train_df, img_dir=self.images_path, transforms=self.train_transform, is_test=False)
    self.val_dataset = FashionDataset(df=val_df, img_dir=self.images_path, transforms=self.val_test_transform, is_test=False)
    self.test_dataset = FashionDataset(df=test_df, img_dir=self.images_path, transforms=self.val_test_transform, is_test=True)

    print(f"Długość całego zbioru: {len(df)}")
    print(f"Długość zbioru treningowego: {len(self.train_dataset)}")
    print(f"Długość zbioru walidacyjnego: {len(self.val_dataset)}")
    print(f"Długość zbioru testowego: {len(self.test_dataset)}")

  def train_dataloader(self):
    train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)
    return train_loader

  def val_dataloader(self):
    val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)
    return val_loader

  def test_dataloader(self):
    test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)
    return test_loader

class FashionClassifier(pl.LightningModule):
  def __init__(self, num_classes, freeze_backbone=True):
    super().__init__()
    self.save_hyperparameters()
    self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    self.backbone = nn.Sequential(*list(self.model.children())[:-1])
    self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    self.criterion = nn.BCEWithLogitsLoss()
    self.accuracy = Accuracy(task="multilabel", num_labels=num_classes, threshold=0.5)
    self.f1 = F1Score(task="multilabel", num_labels=num_classes, threshold=0.5)

    self.pooling = self.model.avgpool

    self.fc1 = nn.Linear(512, 500)
    self.fc2 = nn.Linear(500, num_classes)

    if freeze_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False

    self.examples_for_display = []

  def forward(self, x):
      self.backbone.eval()
      with torch.no_grad():
          x = self.backbone(x)
          x = self.pooling(x).flatten(1)

      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      return x

  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.criterion(logits, y)
    preds = torch.sigmoid(logits)

    acc = self.accuracy(preds, y)
    f1 = self.f1(preds, y)
    self.log('train_loss', loss, prog_bar=True)
    self.log('train_acc', acc, prog_bar=True)
    self.log('train_f1', f1, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    val_loss = self.criterion(logits, y)
    preds = torch.sigmoid(logits)

    val_acc = self.accuracy(preds, y)
    val_f1 = self.f1(preds, y)
    self.log('val_loss', val_loss, prog_bar=True)
    self.log('val_acc', val_acc, prog_bar=True)
    self.log('val_f1', val_f1, prog_bar=True)
    return val_loss

  def test_step(self, batch, batch_idx):
    x, y, paths = batch # usunac paths jesli nie chce wypisac
    logits = self(x)
    test_loss = self.criterion(logits, y)
    preds = torch.sigmoid(logits)

    test_acc = self.accuracy(preds, y)
    test_f1 = self.f1(preds, y)
    self.log('test_loss', test_loss, prog_bar=True)
    self.log('test_acc', test_acc, prog_bar=True)
    self.log('test_f1', test_f1, prog_bar=True)

    if len(self.examples_for_display) < 10: # usunac tego ifa jesli nie chce wypisywac
      preds_np = preds.cpu().numpy()
      y_np = y.cpu().numpy()
      for i in range(len(x)):
        if len(self.examples_for_display) >= 10:
          break
        self.examples_for_display.append({
          "image": paths[i],
          "true": y_np[i],
          "pred": preds_np[i]
        })

    return test_loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer

def predict_single_image(image_input, model_path, mlb, num_classes):

  if isinstance(image_input, str):
    image = Image.open(image_input).convert("RGB")
  else:
    image = image_input.convert("RGB")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = FashionClassifier(num_classes=num_classes)
  state_dict = torch.load(model_path, map_location=device)
  model.load_state_dict(state_dict)
  model.to(device)
  model.eval()

  img_tensor = transform(image).unsqueeze(0).to(device)

  with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

  threshold = 0.3
  labels = [cls for cls, p in zip(mlb.classes_, probs) if p > threshold]

  return labels