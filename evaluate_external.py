import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

MODELS = {
    "Original ResNet-18": Path("models/fake_image_resnet18.pth"),
    "Robust ResNet-18": Path("models/fake_image_resnet18_robust.pth"),
}

EXTERNAL_DIR = Path("external_test")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path):

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# TEST MODEL
# ============================================================

def test_model(model_name, model_path):

    print("\n")
    print("=" * 70)
    print(f"TESTING: {model_name}")
    print("=" * 70)

    print(f"Model: {model_path}")

    model = load_model(model_path)

    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    ai_folder = EXTERNAL_DIR / "ai"
    real_folder = EXTERNAL_DIR / "real"

    ai_images = sorted([
        p for p in ai_folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])

    real_images = sorted([
        p for p in real_folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])

    ai_correct = 0
    real_correct = 0

    # ========================================================
    # AI IMAGES
    # ========================================================

    print("\n" + "-" * 70)
    print("AI IMAGE TEST")
    print("-" * 70)

    for image_path in ai_images:

        try:
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(image)
                probabilities = torch.softmax(output, dim=1)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100

            prediction = "AI" if predicted_class == 0 else "REAL"

            correct = prediction == "AI"

            if correct:
                ai_correct += 1
                mark = "✓"
            else:
                mark = "✗"

            print(
                f"{image_path.name:<15} "
                f"{prediction:<8} "
                f"{confidence:>7.2f}% {mark}"
            )

        except Exception as e:
            print(f"{image_path.name:<15} ERROR: {e}")

    # ========================================================
    # REAL IMAGES
    # ========================================================

    print("\n" + "-" * 70)
    print("REAL IMAGE TEST")
    print("-" * 70)

    for image_path in real_images:

        try:
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(image)
                probabilities = torch.softmax(output, dim=1)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100

            prediction = "AI" if predicted_class == 0 else "REAL"

            correct = prediction == "REAL"

            if correct:
                real_correct += 1
                mark = "✓"
            else:
                mark = "✗"

            print(
                f"{image_path.name:<15} "
                f"{prediction:<8} "
                f"{confidence:>7.2f}% {mark}"
            )

        except Exception as e:
            print(f"{image_path.name:<15} ERROR: {e}")

    # ========================================================
    # RESULTS
    # ========================================================

    total_correct = ai_correct + real_correct
    total_images = len(ai_images) + len(real_images)

    ai_accuracy = (
        ai_correct / len(ai_images) * 100
        if ai_images else 0
    )

    real_accuracy = (
        real_correct / len(real_images) * 100
        if real_images else 0
    )

    overall_accuracy = (
        total_correct / total_images * 100
        if total_images else 0
    )

    print("\n" + "=" * 70)
    print(f"{model_name} RESULTS")
    print("=" * 70)

    print(f"AI Images   : {ai_correct}/{len(ai_images)}")
    print(f"AI Accuracy : {ai_accuracy:.2f}%")

    print(f"Real Images : {real_correct}/{len(real_images)}")
    print(f"Real Accuracy: {real_accuracy:.2f}%")

    print(f"Total       : {total_correct}/{total_images}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    print("=" * 70)

    return {
        "name": model_name,
        "ai_correct": ai_correct,
        "ai_total": len(ai_images),
        "real_correct": real_correct,
        "real_total": len(real_images),
        "total_correct": total_correct,
        "total": total_images,
        "accuracy": overall_accuracy,
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("EXTERNAL AI + REAL IMAGE MODEL COMPARISON")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

results = []

for model_name, model_path in MODELS.items():

    if not model_path.exists():

        print(f"\nWARNING: Model not found: {model_path}")
        continue

    results.append(
        test_model(model_name, model_path)
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':<25}"
    f"{'AI':<12}"
    f"{'Real':<12}"
    f"{'Overall':<12}"
)

print("-" * 70)

for r in results:

    print(
        f"{r['name']:<25}"
        f"{r['ai_correct']}/{r['ai_total']:<9}"
        f"{r['real_correct']}/{r['real_total']:<9}"
        f"{r['accuracy']:.2f}%"
    )

print("=" * 70)