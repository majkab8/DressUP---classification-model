import os
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from classification import (
    load_dataset, build_labels, transform,
    FashionDataset, main_training
)


def train_model():
    torch.multiprocessing.freeze_support()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    df, path = load_dataset()
    df["labels"] = df.apply(build_labels, axis=1)

    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    img_dir = os.path.join(path, "data")

    # Prepare labels
    mlb = MultiLabelBinarizer()
    mlb.fit(train_df["labels"].tolist())

    # Create datasets
    train_dataset = FashionDataset(train_df, img_dir, transform=transform, mlb=mlb)
    val_dataset = FashionDataset(val_df, img_dir, transform=transform, mlb=mlb)
    test_dataset = FashionDataset(test_df, img_dir, transform=transform, mlb=mlb)

    # Train model
    print("Starting training...")
    model = main_training(train_dataset, val_dataset, test_dataset, device)

    # Save model and label encoder
    torch.save(model.state_dict(), "best_model.pth")
    joblib.dump(mlb, "mlb.pkl")

    print("Training complete!")
    print("Saved model as best_model.pth")
    print("Saved label binarizer as mlb.pkl")

    return model, mlb


if __name__ == "__main__":
    train_model()
