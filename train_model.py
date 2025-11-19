import os
import joblib
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import kagglehub

from fashion_classification import (
    train_transform, val_test_transform, FashionDataModule, FashionClassifier
)

def train_model():
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
    print("Path to dataset files:", path)

    dm = FashionDataModule(
        num_classes=127,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        train_transform=train_transform,
        val_test_transform=val_test_transform,
        images_path=os.path.join(path, "fashion-dataset/images"),
        labels_path=os.path.join(path, "fashion-dataset/styles.csv")
    )

    dm.setup()
    model = FashionClassifier(num_classes=dm.num_classes, freeze_backbone=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename="best_model",
        save_weights_only=False
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)

    joblib.dump(dm.mlb, "mlb.pkl")

    best_ckpt_path = checkpoint_callback.best_model_path
    best_model = FashionClassifier.load_from_checkpoint(best_ckpt_path)

    torch.save(best_model.state_dict(), "best_model.pth")

    print(f"\nZapisano najlepszy model do 'best_model.pth'")
    print(f"Zapisano MultiLabelBinarizer do 'mlb.pkl'")

if __name__ == "__main__":
    train_model()
