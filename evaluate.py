import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"E:\kagglehub\datasets\vtphatt2\genimage-adm\versions\1\GenImage\ADM\imagenet_ai_0508_adm\val"

MODEL_PATH = r"E:\fake-image-detector\models\fake_image_resnet18.pth"

BATCH_SIZE = 8
NUM_WORKERS = 0
IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("=" * 60)


# ============================================================
# TRANSFORMS
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
# DATASET
# ============================================================

print("\nLoading validation dataset...")

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

print(f"Validation images: {len(dataset)}")
print(f"Classes: {dataset.classes}")
print(f"Class mapping: {dataset.class_to_idx}")


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading ResNet-18...")

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print(f"\nLoading model from:")
print(MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Handle both checkpoint formats
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
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STARTING EVALUATION")
print("=" * 60)

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(dataloader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Probability of AI/Fake class
        all_probabilities.extend(
            probabilities[:, 0].cpu().numpy()
        )

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Processed: "
                f"{(batch_idx + 1) * BATCH_SIZE}/"
                f"{len(dataset)}"
            )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_prob_fake = np.array(all_probabilities)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=0
)

# For ROC-AUC, convert label 0 (AI) to positive class
y_true_fake = (y_true == 0).astype(int)

auc = roc_auc_score(
    y_true_fake,
    y_prob_fake
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print(f"ROC-AUC  : {auc:.4f}")


print("\n" + "-" * 60)
print("CONFUSION MATRIX")
print("-" * 60)

print("\n                Predicted")
print("              AI       Real")
print(
    f"Actual AI   {cm[0][0]:6d}   {cm[0][1]:6d}"
)
print(
    f"Actual Real {cm[1][0]:6d}   {cm[1][1]:6d}"
)


print("\n" + "-" * 60)
print("CLASSIFICATION REPORT")
print("-" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["AI/Fake", "Real"],
        digits=4
    )
)


print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)