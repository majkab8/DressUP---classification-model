import os
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import joblib


register_heif_opener()

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
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
  def __init__(self, df, img_dir, transforms, encoders, is_test=False):
    self.df = df
    self.img_dir = img_dir
    self.transform = transforms
    self.encoders = encoders
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

    target_type = torch.tensor(self.encoders['type'].transform([row['articleType']])[0], dtype=torch.long)
    target_color = torch.tensor(self.encoders['color'].transform([row['baseColour']])[0], dtype=torch.long)
    target_usage = torch.tensor(self.encoders['usage'].transform([row['usage']])[0], dtype=torch.long)
    target_season = torch.tensor(self.encoders['season'].transform([row['season']])[0], dtype=torch.long)


    if self.is_test:
      return img, target_type, target_color, target_usage, target_season, img_path
    else:
      return img, target_type, target_color, target_usage, target_season

class FashionDataModule(pl.LightningDataModule):
  def __init__(self, batch_size=32, num_workers=0, train_transform=train_transform, val_test_transform=val_test_transform, images_path=None, labels_path=None):
    super().__init__()
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.train_transform = train_transform
    self.val_test_transform = val_test_transform
    self.images_path = images_path
    self.labels_path = labels_path
    self.pin_memory = True
    self.encoders = {}
    self.num_classes = {}

  def setup(self, stage=None):
    df = pd.read_csv(self.labels_path, on_bad_lines='skip')
    important_cols = ['id', 'articleType', 'masterCategory', 'baseColour', 'gender']
    df = df.dropna(subset=important_cols).reset_index(drop=True)

    allowed_categories = ['Apparel', 'Footwear']
    df = df[df['masterCategory'].isin(allowed_categories)]
    exclude_classes = [
      "Socks", "Shoe Accessories", "Shoe Laces",
      "Innerwear Vests", "Trunk", "Boxers", "Briefs", "Bath Robe"
    ]
    df = df[~df['articleType'].isin(exclude_classes)]

    class_counts = df['articleType'].value_counts()
    valid_classes = class_counts[class_counts >= 60].index
    df = df[df['articleType'].isin(valid_classes)].reset_index(drop=True)

    for col in ["season", "baseColour", "subCategory", "articleType", "usage"]:
            if col not in df.columns:
                df[col] = "unknown"
            df[col] = df[col].astype(str).str.lower()

    def check_file_exists(row):
      img_id = str(row['id'])
      for ext in ['.jpg', '.jpeg', '.png', '.heic', '.HEIC']:
        full_path = os.path.join(self.images_path, img_id + ext)
        if os.path.exists(full_path):
          return True
      return False

    df['exists'] = df.apply(check_file_exists, axis=1)
    df = df[df['exists']].reset_index(drop=True)

    self.encoders['type'] = LabelEncoder().fit(df['articleType'])
    self.encoders['color'] = LabelEncoder().fit(df['baseColour'])
    self.encoders['usage'] = LabelEncoder().fit(df['usage'])
    self.encoders['season'] = LabelEncoder().fit(df['season'])

    self.num_classes['type'] = len(self.encoders['type'].classes_)
    self.num_classes['color'] = len(self.encoders['color'].classes_)
    self.num_classes['usage'] = len(self.encoders['usage'].classes_)
    self.num_classes['season'] = len(self.encoders['season'].classes_)

    train_df, rest_df = train_test_split(
      df,
      train_size=0.8,
      shuffle=True,
      random_state=42,
      stratify=df['articleType']
    )

    val_df, test_df = train_test_split(
      rest_df,
      test_size=0.5,
      shuffle=True,
      random_state=42,
      stratify=rest_df['articleType']
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    self.train_dataset = FashionDataset(df=train_df, img_dir=self.images_path, transforms=self.train_transform, encoders=self.encoders, is_test=False)
    self.val_dataset = FashionDataset(df=val_df, img_dir=self.images_path, transforms=self.val_test_transform, encoders=self.encoders, is_test=False)
    self.test_dataset = FashionDataset(df=test_df, img_dir=self.images_path, transforms=self.val_test_transform, encoders=self.encoders, is_test=True)

  def train_dataloader(self):
    train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)
    return train_loader

  def val_dataloader(self):
    val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)
    return val_loader

  def test_dataloader(self):
    test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)
    return test_loader

class FashionClassifier(pl.LightningModule):
  def __init__(self, num_classes_dict, freeze_backbone=False):
    super().__init__()
    self.save_hyperparameters()
    effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    self.backbone = nn.Sequential(
      effnet.features,
      effnet.avgpool
    )
    embedding_size = effnet.classifier[1].in_features

    self.fc_shared = nn.Sequential(
      nn.Flatten(),
      nn.Linear(embedding_size, 512),
      nn.ReLU(),
      nn.Dropout(p=0.5)
    )
    self.criterion = nn.CrossEntropyLoss()


    self.head_type = nn.Linear(512, num_classes_dict['type'])
    self.head_color = nn.Linear(512, num_classes_dict['color'])
    self.head_usage = nn.Linear(512, num_classes_dict['usage'])
    self.head_season = nn.Linear(512, num_classes_dict['season'])

    if freeze_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False

  def forward(self, x):
      x = self.backbone(x)
      x = self.fc_shared(x)

      out_type = self.head_type(x)
      out_color = self.head_color(x)
      out_usage = self.head_usage(x)
      out_season = self.head_season(x)

      return out_type, out_color, out_usage, out_season

  def metrics_calculator(self, outputs, targets, mode='train'):

    output_type, output_color, output_usage, output_season = outputs
    target_type, target_color, target_usage, target_season = targets

    loss_type = self.criterion(output_type, target_type)
    loss_color = self.criterion(output_color, target_color)
    loss_usage = self.criterion(output_usage, target_usage)
    loss_season = self.criterion(output_season, target_season)
    total = loss_type + loss_color + loss_usage + loss_season

    acc_type = (output_type.argmax(dim=1) == target_type).float().mean()
    acc_color = (output_color.argmax(dim=1) == target_color).float().mean()
    acc_usage = (output_usage.argmax(dim=1) == target_usage).float().mean()
    acc_season = (output_season.argmax(dim=1) == target_season).float().mean()

    self.log(f'{mode}_loss', total, prog_bar=True)
    self.log(f'{mode}_acc_type', acc_type, prog_bar=True)
    self.log(f'{mode}_acc_color', acc_color, prog_bar=True)
    self.log(f'{mode}_acc_usage', acc_usage, prog_bar=True)
    self.log(f'{mode}_acc_season', acc_season, prog_bar=True)

    return total

  def training_step(self, batch, batch_idx):
    img, train_type, train_color, train_usage, train_season = batch
    outputs = self(img)

    return self.metrics_calculator(outputs, (train_type, train_color, train_usage, train_season), mode='train')

  def validation_step(self, batch, batch_idx):
    img, train_type, train_color, train_usage, train_season = batch
    outputs = self(img)

    return self.metrics_calculator(outputs, (train_type, train_color, train_usage, train_season), mode='val')

  def test_step(self, batch, batch_idx):
    img, train_type, train_color, train_usage, train_season, paths = batch
    outputs = self(img)

    return self.metrics_calculator(outputs, (train_type, train_color, train_usage, train_season), mode='test')

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
    return optimizer

def plot_model_results(log_dir):
  metrics_path = os.path.join(log_dir, "metrics.csv")
  if not os.path.exists(metrics_path):
    print("Brak pliku metrics.csv")
    return

  metrics = pd.read_csv(metrics_path)

  epoch_metrics = metrics.groupby("epoch").mean(numeric_only=True)

  fig, axes = plt.subplots(2, 2, figsize=(15, 10))
  fig.suptitle('Dokładność (Accuracy) dla poszczególnych cech', fontsize=16)

  heads = [
    ('type', 'Typ Ubrania', axes[0, 0]),
    ('color', 'Kolor', axes[0, 1]),
    ('usage', 'Okazja', axes[1, 0]),
    ('season', 'Sezon', axes[1, 1])
  ]

  for key, title, ax in heads:
    train_key = f'train_acc_{key}'
    val_key = f'val_acc_{key}'

    if train_key in epoch_metrics:
      ax.plot(epoch_metrics.index, epoch_metrics[train_key], label='Train', marker='.')

    if val_key in epoch_metrics:
      ax.plot(epoch_metrics.index, epoch_metrics[val_key], label='Val', marker='.')

    ax.set_title(title)
    ax.set_xlabel('Epoka')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1.0])
    ax.grid(True)
    ax.legend()

  plt.tight_layout()
  plt.savefig("multihead_training_results.png")
  plt.show()

def plot_confusion_matrix(model, datamodule):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()

  y_true = []
  y_pred = []

  loader = datamodule.test_dataloader()

  with torch.no_grad():
    for batch in loader:
      if len(batch) == 6:
        imgs, true_type, *reszta = batch
      else:
        imgs, true_type, *reszta = batch

      out_type, _, _, _ = model(imgs.to(device))
      preds = out_type.argmax(dim=1).cpu().numpy()

      y_true.extend(true_type.numpy())
      y_pred.extend(preds)

  class_names = datamodule.encoders['type'].classes_

  fig, ax = plt.subplots(figsize=(16, 16))

  disp = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=class_names,
    cmap='Blues',
    ax=ax,
    xticks_rotation='vertical'
  )

  disp.ax_.set_title("Confusion matrix")

  plt.tight_layout()
  plt.savefig("confusion_matrix.png")
  plt.show()


def predict_single_image(image_input, model_path="best_model.pth", encoders_path="encoders.pkl"):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  encoders = joblib.load(encoders_path)
  num_classes = {k: len(v.classes_) for k, v in encoders.items()}

  model = FashionClassifier(num_classes_dict=num_classes)
  state_dict = torch.load(model_path, map_location=device)
  model.load_state_dict(state_dict)
  model.to(device)
  model.eval()

  if isinstance(image_input, str):
    image = Image.open(image_input).convert("RGB")
  else:
    image = image_input.convert("RGB")

  transform = val_test_transform
  img_tensor = transform(image).unsqueeze(0).to(device)

  with torch.no_grad():
    o_type, o_color, o_usage, o_season = model(img_tensor)

    pred_type = torch.argmax(o_type, dim=1).item()
    pred_color = torch.argmax(o_color, dim=1).item()
    pred_usage = torch.argmax(o_usage, dim=1).item()
    pred_season = torch.argmax(o_season, dim=1).item()

  results = {
    "Type": encoders['type'].inverse_transform([pred_type])[0],
    "Color": encoders['color'].inverse_transform([pred_color])[0],
    "Usage": encoders['usage'].inverse_transform([pred_usage])[0],
    "Season": encoders['season'].inverse_transform([pred_season])[0]
  }
  return results
