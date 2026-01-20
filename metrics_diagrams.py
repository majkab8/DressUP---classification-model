import os
import torch
import joblib
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from fashion_classification import FashionDataModule, FashionClassifier

def evaluate_model_comprehensive(model, datamodule):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = {k: [] for k in ['type', 'color', 'usage', 'season']}
    all_targets = {k: [] for k in ['type', 'color', 'usage', 'season']}

    loader = datamodule.test_dataloader()

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                imgs, t_type, t_color, t_usage, t_season, _ = batch
            else:
                imgs, t_type, t_color, t_usage, t_season = batch

            out_type, out_color, out_usage, out_season = model(imgs.to(device))

            all_preds['type'].extend(out_type.argmax(dim=1).cpu().numpy())
            all_targets['type'].extend(t_type.numpy())

            all_preds['color'].extend(out_color.argmax(dim=1).cpu().numpy())
            all_targets['color'].extend(t_color.numpy())

            all_preds['usage'].extend(out_usage.argmax(dim=1).cpu().numpy())
            all_targets['usage'].extend(t_usage.numpy())

            all_preds['season'].extend(out_season.argmax(dim=1).cpu().numpy())
            all_targets['season'].extend(t_season.numpy())

    heads_info = [
        ('type', 'Typ ubrania'),
        ('color', 'Kolor'),
        ('usage', 'Okazja'),
        ('season', 'Sezon')
    ]

    sns.set_theme(style="white")

    for key, title in heads_info:
        y_true = all_targets[key]
        y_pred = all_preds[key]

        class_names = datamodule.encoders[key].classes_

        acc = accuracy_score(y_true, y_pred)

        all_labels = np.arange(len(class_names))

        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Macierz pomyłek - {title}\nDokładność: {acc:.2%}', fontsize=16)
        plt.xlabel('Przewidziane')
        plt.ylabel('Prawdziwe')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"matrix_{key}.png")
        plt.close()

        report = classification_report(
            y_true,
            y_pred,
            labels=all_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        df = pd.DataFrame(report).transpose()
        df = df.drop(['accuracy', 'macro avg', 'weighted avg'])
        df = df.sort_values(by='f1-score', ascending=False)

        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(1, 3, figsize=(20, len(class_names) * 0.4 + 4))
        fig.suptitle(f'Metryki: {title}', fontsize=20, y=1.02)

        sns.barplot(ax=axes[0], x=df['precision'], y=df.index, color='skyblue')
        axes[0].set_title('Precyzja', fontsize=14)
        axes[0].set_xlim(0, 1.1)
        axes[0].axvline(0.5, color='red', linestyle='--', alpha=0.3)

        sns.barplot(ax=axes[1], x=df['recall'], y=df.index, color='lightgreen')
        axes[1].set_title('Czułość', fontsize=14)
        axes[1].set_xlim(0, 1.1)
        axes[1].set_yticks([])
        axes[1].axvline(0.5, color='red', linestyle='--', alpha=0.3)

        sns.barplot(ax=axes[2], x=df['f1-score'], y=df.index, color='plum')
        axes[2].set_title('F1-Score', fontsize=14)
        axes[2].set_xlim(0, 1.1)
        axes[2].set_yticks([])

        for i, v in enumerate(df['f1-score']):
            axes[2].text(v + 0.02, i, f"{v:.2f}", color='black', va='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"metrics_{key}.png", bbox_inches='tight')
        plt.close()

        sns.set_theme(style="white")

if __name__ == "__main__":
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
    images_path = os.path.join(path, "fashion-dataset/images")
    labels_path = os.path.join(path, "fashion-dataset/styles.csv")

    try:
        encoders = joblib.load("encoders.pkl")
    except FileNotFoundError:
        print("File encoders.pkl not found")
        exit()

    num_classes = {
        'type': len(encoders['type'].classes_),
        'color': len(encoders['color'].classes_),
        'usage': len(encoders['usage'].classes_),
        'season': len(encoders['season'].classes_)
    }

    dm = FashionDataModule(
        images_path=images_path,
        labels_path=labels_path,
        batch_size=32,
        num_workers=2
    )
    dm.setup()

    dm.encoders = encoders

    model = FashionClassifier(num_classes_dict=num_classes)

    model_path = "best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model loaded.")
    else:
        print(f"Model file not found")
        exit()

    evaluate_model_comprehensive(model, dm)