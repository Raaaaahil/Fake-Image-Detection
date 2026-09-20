import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fake_image_resnet18_robust.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load ResNet-18
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()


def predict_image(image: Image.Image):

    image = image.convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

    ai_probability = probabilities[0][0].item()
    real_probability = probabilities[0][1].item()

    if ai_probability > real_probability:
        prediction = "AI Generated"
        confidence = ai_probability * 100
        predicted_class = "AI"
    else:
        prediction = "Real / Natural"
        confidence = real_probability * 100
        predicted_class = "REAL"

    return {
        "prediction": prediction,
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "ai_probability": round(ai_probability * 100, 2),
        "real_probability": round(real_probability * 100, 2)
    }