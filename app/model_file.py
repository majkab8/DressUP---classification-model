import torch
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class FashionClassifierInference(nn.Module):
    def __init__(self, num_classes_dict):
        super().__init__()
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
            nn.Dropout(0.5)
        )

        self.head_type = nn.Linear(512, num_classes_dict['type'])
        self.head_color = nn.Linear(512, num_classes_dict['color'])
        self.head_usage = nn.Linear(512, num_classes_dict['usage'])
        self.head_season = nn.Linear(512, num_classes_dict['season'])

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_shared(x)

        return self.head_type(x), self.head_color(x), self.head_usage(x), self.head_season(x)

class FashionModel:
    _instance = None

    def __new__(cls, model_path, encoders):
        if cls._instance is None:
            cls._instance = super(FashionModel, cls).__new__(cls)
            cls._instance._init(model_path, encoders)
        return cls._instance

    def _init(self, model_path, encoders):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoders = encoders

        self.num_classes_dict = {
            'type': len(encoders['type'].classes_),
            'color': len(encoders['color'].classes_),
            'usage': len(encoders['usage'].classes_),
            'season': len(encoders['season'].classes_)
        }

        self.model = FashionClassifierInference(num_classes_dict=self.num_classes_dict)

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
            o_type, o_color, o_usage, o_season = self.model(img_tensor)

            pred_type_idx = torch.argmax(o_type, dim=1).item()
            pred_color_idx = torch.argmax(o_color, dim=1).item()
            pred_usage_idx = torch.argmax(o_usage, dim=1).item()
            pred_season_idx = torch.argmax(o_season, dim=1).item()

        results = {
            "type": self.encoders['type'].inverse_transform([pred_type_idx])[0],
            "color": self.encoders['color'].inverse_transform([pred_color_idx])[0],
            "usage": self.encoders['usage'].inverse_transform([pred_usage_idx])[0],
            "season": self.encoders['season'].inverse_transform([pred_season_idx])[0]
        }

        return results
