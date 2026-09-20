import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(
    r"E:\kagglehub\datasets\vtphatt2\genimage-adm\versions\1"
    r"\GenImage\ADM\imagenet_ai_0508_adm"
)

MODEL_PATH = Path("models/fake_image_resnet18.pth")
OUTPUT_PATH = Path("models/fake_image_resnet18_robust.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

# Start with only 3 epochs.
# We can extend if the result is promising.
NUM_EPOCHS = 3

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ROBUST AUGMENTATION
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0),
        interpolation=InterpolationMode.BILINEAR
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05
        )
    ], p=0.5),

    transforms.RandomGrayscale(p=0.05),

    transforms.RandomApply([
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 2.0)
        )
    ], p=0.2),

    transforms.ToTensor(),

    transforms.RandomErasing(
        p=0.10,
        scale=(0.02, 0.10),
        ratio=(0.3, 3.3)
    ),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose([
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

print("=" * 65)
print("ROBUST RESNET-18 FINE-TUNING")
print("=" * 65)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading training dataset...")

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Classes: {train_dataset.class_to_idx}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# ============================================================
# MODEL
# ============================================================

print("\nLoading your 99.91% ResNet-18...")

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

print("Baseline model loaded successfully.")

# ============================================================
# OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = DEVICE.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)

# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=use_amp
            ):

                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        correct / total * 100
    )

# ============================================================
# TRAINING
# ============================================================

best_val_acc = 0.0

print("\nStarting robust fine-tuning...")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("=" * 65)

for epoch in range(NUM_EPOCHS):

    start_time = time.time()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if (batch_idx + 1) % 5000 == 0:

            print(
                f"  Batch {batch_idx + 1}/"
                f"{len(train_loader)}"
            )

    train_loss = running_loss / total
    train_acc = correct / total * 100

    val_loss, val_acc = validate()

    elapsed = (time.time() - start_time) / 60

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    print(
        f"Time: {elapsed:.2f} minutes"
    )

    # Save best robust model
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "val_accuracy": val_acc,
                "epoch": epoch + 1
            },
            OUTPUT_PATH
        )

        print(
            f"✓ Saved new best model: "
            f"{best_val_acc:.2f}%"
        )

print("\n" + "=" * 65)
print("ROBUST FINE-TUNING COMPLETE")
print("=" * 65)

print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Model saved to: {OUTPUT_PATH}")