import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class FashionClassifier(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.pooling = m.avgpool
        self.fc1 = nn.Linear(512, 500)
        self.fc2 = nn.Linear(500, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
            x = self.pooling(x).flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def load_model(model_path: str, num_classes: int, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FashionClassifier(num_classes=num_classes)
    state = torch.load(model_path, map_location=device)

    if any(k.startswith("conv1.") for k in state.keys()):
        print("Detected raw ResNet18 checkpoint. Loading into backbone...")
        model_state = model.state_dict()
        matched_state = {k: v for k, v in state.items() if k in model_state}
        model_state.update(matched_state)
        model.load_state_dict(model_state)
    else:
        print("Detected FashionClassifier checkpoint. Loading normally...")
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model


def predict_single_image(image: Image.Image, model_path: str, mlb, num_classes: int, threshold: float = 0.3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_path, num_classes=num_classes, device=device)

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    labels = [cls for cls, p in zip(mlb.classes_, probs) if p > threshold]
    return labels
