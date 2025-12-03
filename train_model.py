import os
import joblib
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import kagglehub
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger
import pandas as pd

from fashion_classification import (
    train_transform, val_test_transform, FashionDataModule, FashionClassifier
)

torch.set_float32_matmul_precision('medium')

def train_model():
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
    print("Path to dataset files:", path)

    logger = CSVLogger("logs", name="fashion_model")

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
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger,
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

    plot_training_results(logger.log_dir)


def plot_training_results(log_dir):
    metrics_path = os.path.join(log_dir, "metrics.csv")

    if not os.path.exists(metrics_path):
        print("Nie znaleziono pliku metrics.csv")
        return

    metrics = pd.read_csv(metrics_path)

    epoch_metrics = metrics.groupby("epoch").mean(numeric_only=True)

    test_metrics = None
    if 'test_loss' in metrics.columns:
        test_rows = metrics.dropna(subset=['test_loss'])
        if not test_rows.empty:
            test_metrics = test_rows.iloc[-1]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    if 'train_loss' in epoch_metrics: plt.plot(epoch_metrics['train_loss'], label='Train Loss')
    if 'val_loss' in epoch_metrics: plt.plot(epoch_metrics['val_loss'], label='Val Loss')
    if test_metrics is not None and 'test_loss' in test_metrics:
        plt.axhline(y=test_metrics['test_loss'], color='r', linestyle='--',
                    label=f'Test: {test_metrics["test_loss"]:.3f}')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    if 'train_acc' in epoch_metrics: plt.plot(epoch_metrics['train_acc'], label='Train Acc')
    if 'val_acc' in epoch_metrics: plt.plot(epoch_metrics['val_acc'], label='Val Acc')
    if test_metrics is not None and 'test_acc' in test_metrics:
        plt.axhline(y=test_metrics['test_acc'], color='r', linestyle='--',
                    label=f'Test: {test_metrics["test_acc"]:.3f}')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    if 'train_f1' in epoch_metrics: plt.plot(epoch_metrics['train_f1'], label='Train F1')
    if 'val_f1' in epoch_metrics: plt.plot(epoch_metrics['val_f1'], label='Val F1')
    if test_metrics is not None and 'test_f1' in test_metrics:
        plt.axhline(y=test_metrics['test_f1'], color='r', linestyle='--', label=f'Test: {test_metrics["test_f1"]:.3f}')
    plt.title("F1 Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    save_path = "training_results.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Wykresy zapisano w {save_path}")
    plt.show()

if __name__ == "__main__":
    train_model()
