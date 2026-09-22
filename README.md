# DressUP – Fashion Classification Model

Multi-head image classifier that tags clothing photos with **type**, **colour**, **usage** and **season**,
served as a FastAPI endpoint and deployed to Azure Container Apps.

The model is the AI tagging component of the DressUP application: it takes an image of a garment
and returns a structured set of labels that the rest of the system can use for search and filtering.

---

## Model

| | |
|---|---|
| Backbone | EfficientNet-B0 (ImageNet weights, fully fine-tuned) |
| Head | Shared `Linear(1280 → 512) + ReLU + Dropout(0.5)`, then 4 independent classification heads |
| Loss | Sum of four `CrossEntropyLoss` terms (one per head) |
| Optimiser | Adam, `lr=1e-4`, `weight_decay=1e-4` |
| Input | RGB 256×256, ImageNet normalisation |
| Framework | PyTorch Lightning |

Instead of one multi-label output, each attribute has its own softmax head, so every prediction
returns exactly one label per category.

### Label space

90 labels across four heads (full list in [`tags_list.txt`](tags_list.txt)):

| Head | Classes | Examples |
|---|---|---|
| `type` | 32 | tshirts, jeans, sarees, heels, sweaters |
| `color` | 45 | black, navy blue, olive, grey melange |
| `usage` | 8 | casual, formal, ethnic, sports, party |
| `season` | 5 | spring, summer, fall, winter |

## Dataset

[Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
(Kaggle, ~44k images), downloaded automatically via `kagglehub`. Preprocessing in
`FashionDataModule.setup()`:

- keeps only the `Apparel` and `Footwear` master categories,
- drops rows with missing `id`, `articleType`, `masterCategory`, `baseColour` or `gender`,
- excludes underwear / shoe-accessory classes and any class with fewer than 60 samples,
- lowercases all label columns and fits a `LabelEncoder` per head,
- stratified 80 / 10 / 10 split on `articleType` (`random_state=42`).

Training augmentation: random horizontal flip and mild colour jitter. HEIC images are supported
through `pillow-heif`.

## Results

Test-set accuracy of the shipped checkpoint (`best_model.pth`, run `logs/fashion_model/version_11`):

| Head | Accuracy |
|---|---|
| Usage | 93.1 % |
| Type | 89.2 % |
| Season | 80.0 % |
| Colour | 74.6 % |

Per-class precision / recall / F1 and confusion matrices are in `metrics_*.png` and `matrix_*.png`;
the training curves are in `multihead_training_results.png`. Colour is the weakest head — most of its
errors are confusions between neighbouring shades (e.g. *navy blue* vs *blue*, *grey* vs *grey melange*).

---

## Repository layout

```
fashion_classification.py   Dataset, LightningDataModule, LightningModule, single-image prediction
train_model.py              Training entry point (early stopping + best-checkpoint saving)
metrics_diagrams.py         Test-set evaluation: confusion matrices and per-class metric plots
counting_tags.py            Dumps the encoder classes to tags_list.txt
best_model.pth              Trained weights
encoders.pkl                Fitted LabelEncoders (one per head)
app/                        FastAPI inference service + Dockerfile + Azure deployment guide
logs/                       CSVLogger output from training runs
```

## Setup

Requires Python 3.10+. A CUDA GPU is recommended for training but not required for inference.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the CUDA 12.6 builds of `torch` / `torchvision`; install the CPU wheels
instead if you have no GPU.

## Training

```bash
python train_model.py
```

Downloads the dataset, trains for up to 25 epochs with early stopping on `val_loss`
(patience 5), then writes `best_model.pth` and `encoders.pkl` and logs metrics to
`logs/fashion_model/version_N/metrics.csv`.

## Evaluation

```bash
python metrics_diagrams.py
```

Loads `best_model.pth` + `encoders.pkl`, runs the test split and regenerates
`matrix_<head>.png` and `metrics_<head>.png`.

## Inference

Single image, from Python:

```python
from fashion_classification import predict_single_image

predict_single_image("shirt.jpg")
# {'Type': 'shirts', 'Color': 'navy blue', 'Usage': 'casual', 'Season': 'summer'}
```

## API

The service in `app/` exposes one endpoint. It expects `best_model.pth` and `encoders.pkl`
next to `app.py`, and loads the model once as a singleton.

```bash
cp best_model.pth encoders.pkl app/
cd app
uvicorn app:app --host 0.0.0.0 --port 8000
```

**`POST /predict_batch`** — takes a list of image URLs, downloads each one and returns its tags.
Failures are reported per URL instead of failing the whole batch.

```jsonc
// request
{ "urls": ["https://example.com/shirt.jpg"] }

// response
{
  "results": [
    {
      "url": "https://example.com/shirt.jpg",
      "tags": { "type": "shirts", "color": "navy blue", "usage": "casual", "season": "summer" }
    }
  ]
}
```

## Deployment

Containerised with the `app/Dockerfile` (CPU-only torch build) and deployed to Azure Container Apps
via Azure Container Registry. Step-by-step commands: [`app/readme.md`](app/readme.md).
