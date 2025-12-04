import torch
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch.nn.functional as F


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class FashionClassifierInference(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.model.children())[:-1])
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.pooling = self.model.avgpool

        self.fc1 = nn.Linear(512, 500)
        self.fc2 = nn.Linear(500, num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x).flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class FashionModel:
    _instance = None

    def __new__(cls, model_path, mlb, num_classes, threshold=0.5):
        if cls._instance is None:
            cls._instance = super(FashionModel, cls).__new__(cls)
            cls._instance._init(model_path, mlb, num_classes, threshold)
        return cls._instance

    def _init(self, model_path, mlb, num_classes, threshold):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.threshold = threshold
        self.mlb = mlb

        self.model = FashionClassifierInference(num_classes=num_classes, freeze_backbone=False)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        img_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        labels = [cls for cls, p in zip(self.mlb.classes_, probs) if p > self.threshold]
        return labels
