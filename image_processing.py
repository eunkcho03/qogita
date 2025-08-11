import io
import requests
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


class ImageEmbedder:
    def __init__(self, device=None, timeout=5.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout = timeout
        
        # Load pre-trained ResNet18 and replace final layer with identity
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()  # Output 512-d vector
        self.model = self.model.to(self.device).eval()

        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def embed_one(self, url):
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Invalid URL.")
        
        # Download and preprocess image
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        
        # Extract features
        feat = self.model(x).cpu().squeeze(0)  # [512]
        return feat


    def embed_dataframe(self, df, url_column="Image URL"):
        urls = df[url_column].tolist()
        feats = [self.embed_one(u) for u in urls]
        return torch.stack(feats, dim=0)

