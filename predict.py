import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"E:\fake-image-detector\models\fake_image_resnet18.pth"
IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("AI IMAGE DETECTOR")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading ResNet-18...")

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# GET IMAGE PATH
# ============================================================

if len(sys.argv) < 2:

    print("\nUsage:")
    print("python predict.py <image_path>")

    print("\nExample:")
    print(
        r"python predict.py E:\fake-image-detector\test_images\photo.jpg"
    )

    sys.exit()


image_path = sys.argv[1]


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(image_path):

    print(f"\nERROR: Image not found:")
    print(image_path)

    sys.exit()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        image_path
    ).convert("RGB")

except Exception as e:

    print("\nERROR: Could not open image.")
    print(e)

    sys.exit()


# ============================================================
# PREPROCESS
# ============================================================

input_tensor = transform(
    image
).unsqueeze(0).to(device)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(
        input_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()


# ============================================================
# RESULTS
# ============================================================

ai_probability = probabilities[0].item()
real_probability = probabilities[1].item()

if predicted_class == 0:

    prediction = "AI GENERATED / FAKE"
    confidence = ai_probability

else:

    prediction = "REAL / NATURAL"
    confidence = real_probability


print("\n")
print("=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(f"\nImage:")
print(image_path)

print(f"\nPrediction:")
print(prediction)

print(f"\nConfidence:")
print(f"{confidence * 100:.2f}%")

print("\nClass probabilities:")
print(
    f"AI Generated : {ai_probability * 100:.2f}%"
)
print(
    f"Real         : {real_probability * 100:.2f}%"
)

print("=" * 60)