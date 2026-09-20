import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = r"E:\kagglehub\datasets\vtphatt2\genimage-adm\versions\1\GenImage\ADM\imagenet_ai_0508_adm"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

MODEL_DIR = "models"

BATCH_SIZE = 8
NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )

print("=" * 60)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"Validation directory not found:\n{VAL_DIR}"
    )

print("\nDataset found successfully.")
print("Train:", TRAIN_DIR)
print("Validation:", VAL_DIR)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

# ============================================================
# REMOVE CORRUPTED / UNREADABLE IMAGES
# ============================================================

from PIL import Image

def remove_corrupted_images(dataset, dataset_name):

    valid_samples = []
    corrupted_files = []

    print(
        f"\nChecking {dataset_name} images for corruption..."
    )

    for path, label in tqdm(
        dataset.samples,
        desc=f"Checking {dataset_name}"
    ):

        try:
            # First verification
            with Image.open(path) as img:
                img.verify()

            # Re-open to make sure it can actually be decoded
            with Image.open(path) as img:
                img.convert("RGB")

            valid_samples.append(
                (path, label)
            )

        except Exception as e:

            corrupted_files.append(
                (path, str(e))
            )

    # Replace dataset samples with valid files
    dataset.samples = valid_samples

    dataset.targets = [
        label for path, label in valid_samples
    ]

    print(
        f"\n{dataset_name} dataset check complete."
    )

    print(
        f"Valid images: {len(valid_samples)}"
    )

    print(
        f"Corrupted images: {len(corrupted_files)}"
    )

    if corrupted_files:

        print("\nCorrupted files:")

        for path, error in corrupted_files[:20]:

            print(
                f"  {path}"
            )

        if len(corrupted_files) > 20:

            print(
                f"  ... and "
                f"{len(corrupted_files) - 20} more"
            )

    return corrupted_files


train_corrupted = remove_corrupted_images(
    train_dataset,
    "Training"
)

val_corrupted = remove_corrupted_images(
    val_dataset,
    "Validation"
)

print("\nDataset information:")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

print("Classes:", train_dataset.classes)
print("Class mapping:", train_dataset.class_to_idx)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

from collections import Counter

train_counts = Counter(train_dataset.targets)
val_counts = Counter(val_dataset.targets)

print("\nTraining class distribution:")

for class_name, class_index in train_dataset.class_to_idx.items():
    print(
        f"{class_name}:",
        train_counts[class_index]
    )

print("\nValidation class distribution:")

for class_name, class_index in val_dataset.class_to_idx.items():
    print(
        f"{class_name}:",
        val_counts[class_index]
    )


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading ResNet-18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# Replace final classification layer
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_resnet18.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("\nLoaded Epoch 4 checkpoint.")
print(
    f"Previous best validation accuracy: "
    f"{checkpoint['val_accuracy']:.2f}%"
)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)


# ============================================================
# MIXED PRECISION
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 99.78

best_model_weights = copy.deepcopy(
    model.state_dict()
)

os.makedirs(MODEL_DIR, exist_ok=True)


print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(9, NUM_EPOCHS):

    start_time = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    for images, labels in progress_bar:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = (
        100.0 * correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            val_loss_total += loss.item()

            _, predictions = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()


    val_loss = (
        val_loss_total /
        len(val_loader)
    )

    val_accuracy = (
        100.0 * val_correct / val_total
    )

    scheduler.step(val_loss)


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_to_idx":
                    train_dataset.class_to_idx,

                "val_accuracy":
                    val_accuracy,

                "epoch":
                    epoch + 1
            },
            os.path.join(
                MODEL_DIR,
                "best_resnet18.pth"
            )
        )


    elapsed = time.time() - start_time


    print("\n")
    print("-" * 60)

    print(
        f"Epoch: {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    print(
        f"Best Val Accuracy: {best_val_accuracy:.2f}%"
    )

    print(
        f"Time: {elapsed / 60:.2f} minutes"
    )

    print("-" * 60)


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)


# ============================================================
# FINAL SAVE
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "class_to_idx":
            train_dataset.class_to_idx,

        "best_val_accuracy":
            best_val_accuracy
    },
    os.path.join(
        MODEL_DIR,
        "fake_image_resnet18.pth"
    )
)


print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Model saved to:"
)

print(
    os.path.abspath(
        os.path.join(
            MODEL_DIR,
            "fake_image_resnet18.pth"
        )
    )
)

print("=" * 60)