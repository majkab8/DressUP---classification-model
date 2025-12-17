import os
import joblib
import pytorch_lightning as pl
import torch
import kagglehub
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from fashion_classification import (
    FashionDataModule, FashionClassifier, plot_model_results
)

torch.set_float32_matmul_precision('medium')

def train_model():
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")

    logger = CSVLogger("logs", name="fashion_model")

    dm = FashionDataModule(
        images_path=os.path.join(path, "fashion-dataset/images"),
        labels_path=os.path.join(path, "fashion-dataset/styles.csv"),
        batch_size=32,
        num_workers=2
    )

    dm.setup()
    model = FashionClassifier(num_classes_dict=dm.num_classes)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best_model"
    )

    trainer = pl.Trainer(
        max_epochs=25,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        log_every_n_steps=10
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)

    plot_model_results(logger.log_dir)
    torch.save(model.state_dict(), "best_model.pth")
    joblib.dump(dm.encoders, "encoders.pkl")

if __name__ == "__main__":
    train_model()
