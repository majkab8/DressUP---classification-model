# DressUP Classification API

Simple API for predicting fashion tags from images using a trained ResNet model.

## Requirements

- Python 3.10+
- GPU recommended but optional
- Docker (optional, if using container)

## Install Dependencies


```bash
python -m venv .venv

.venv\Scripts\activate

python.exe -m pip install --upgrade pip

python.exe -m pip install -r requirements.txt
```

Launch Locally

Make sure your trained model and label binarizer exist:

```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```