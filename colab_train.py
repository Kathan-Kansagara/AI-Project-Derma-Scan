# ============================================================================
#  DermaScan AI — Google Colab GPU Training Script
#  Copy-paste each section into separate Colab cells
# ============================================================================

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: Setup & Install Dependencies                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# !pip install -q torch torchvision scikit-learn matplotlib seaborn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os, json, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("⚠️  No GPU — training will be slow. Go to Runtime > Change runtime type > GPU")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2: Upload Dataset                                                 ║
# ║  Option A: Upload dataset.zip from your PC                              ║
# ║  Option B: Mount Google Drive if dataset is there                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --- Option A: Upload ZIP from PC ---
# from google.colab import files
# uploaded = files.upload()  # Select your dataset.zip
# !unzip -q dataset.zip -d /content/dataset

# --- Option B: Google Drive ---
# from google.colab import drive
# drive.mount('/content/drive')
# !cp /content/drive/MyDrive/dataset.zip /content/
# !unzip -q /content/dataset.zip -d /content/dataset

# --- Set your data path here ---
DATA_DIR = "/content/dataset/train"  # Adjust if your structure differs

# Verify dataset
if os.path.exists(DATA_DIR):
    classes = sorted(os.listdir(DATA_DIR))
    classes = [c for c in classes if os.path.isdir(os.path.join(DATA_DIR, c))]
    total = sum(len(os.listdir(os.path.join(DATA_DIR, c))) for c in classes)
    print(f"✅ Dataset found: {total} images across {len(classes)} classes")
    for i, c in enumerate(classes):
        count = len(os.listdir(os.path.join(DATA_DIR, c)))
        print(f"   {i+1:2d}. {c[:50]:50s} — {count} images")
else:
    print(f"❌ Dataset not found at {DATA_DIR}")
    print("   Upload your dataset.zip and unzip it first!")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3: Model & Training Components                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class FocalLoss(nn.Module):
    """Handles class imbalance better than standard CrossEntropy."""
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def build_model(num_classes, dropout=0.3):
    """EfficientNet-B0 with custom classifier head."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_feat, num_classes)
    )
    return model


def get_transforms(is_training, img_size=224):
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def compute_class_weights(dataset):
    counts = np.bincount(dataset.targets)
    total = counts.sum()
    weights = total / (len(counts) * counts.astype(float))
    weights = np.clip(weights, 0.5, 5.0)
    return torch.FloatTensor(weights)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4: TRAIN — Run this cell to start training                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# === HYPERPARAMETERS (tune these) ===
EPOCHS = 20           # More epochs on GPU (was 5 on CPU)
FREEZE_EPOCHS = 2     # Head warm-up phase
BATCH_SIZE = 64       # GPU can handle larger batches
LR_HEAD = 1e-3        # Learning rate for head warm-up
LR_FINETUNE = 5e-4    # Learning rate for full fine-tuning
PATIENCE = 5          # Early stopping patience
IMG_SIZE = 224
NUM_WORKERS = 2

print("=" * 70)
print("  DermaScan AI — GPU Training Pipeline")
print("=" * 70)

# Load dataset
print(f"\n[1/6] Loading dataset from '{DATA_DIR}'...")
full_ds = datasets.ImageFolder(DATA_DIR, transform=get_transforms(True, IMG_SIZE))
num_classes = len(full_ds.classes)
print(f"  → {len(full_ds)} images, {num_classes} classes")

# Stratified split
print("\n[2/6] Stratified 80/20 split...")
targets = np.array(full_ds.targets)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))

train_set = Subset(full_ds, train_idx)
val_ds = datasets.ImageFolder(DATA_DIR, transform=get_transforms(False, IMG_SIZE))
val_set = Subset(val_ds, val_idx)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
print(f"  Train: {len(train_set)} | Val: {len(val_set)}")

# Class weights & loss
print("\n[3/6] Computing class weights...")
class_weights = compute_class_weights(full_ds).to(device)
criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.1)

# Model
print(f"\n[4/6] Building EfficientNet-B0 on {device}...")
model = build_model(num_classes).to(device)

# === TRAINING LOOP ===
best_val_acc = 0.0
best_val_loss = float('inf')
no_improve = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
total_epochs = FREEZE_EPOCHS + EPOCHS

# Freeze backbone initially
for p in model.features.parameters():
    p.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR_HEAD, weight_decay=1e-4)
scheduler = None

print(f"\n[5/6] Training ({total_epochs} total epochs)...\n")

for epoch in range(1, total_epochs + 1):
    t0 = time.time()

    # Unfreeze at transition
    if epoch == FREEZE_EPOCHS + 1:
        print(f"\n  ── Unfreezing backbone for fine-tuning ──\n")
        for p in model.features.parameters():
            p.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR_FINETUNE,
            steps_per_epoch=len(train_loader), epochs=EPOCHS
        )
        no_improve = 0

    # --- Train ---
    model.train()
    t_loss, t_correct, t_total = 0, 0, 0

    for batch_i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if epoch > FREEZE_EPOCHS:
            mixed, ya, yb, lam = mixup_data(imgs, labels)
            out = model(mixed)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
            _, pred = out.max(1)
            t_correct += (lam * pred.eq(ya).sum() + (1-lam) * pred.eq(yb).sum()).item()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            _, pred = out.max(1)
            t_correct += pred.eq(labels).sum().item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        t_loss += loss.item() * imgs.size(0)
        t_total += labels.size(0)

        # Progress bar
        pct = (batch_i + 1) / len(train_loader)
        bar = '█' * int(30 * pct) + '·' * (30 - int(30 * pct))
        print(f"\r  Epoch {epoch:2d}/{total_epochs} [{bar}] "
              f"Loss:{loss.item():.4f} Acc:{100*t_correct/t_total:.1f}%", end="")

    t_loss /= t_total
    t_acc = 100 * t_correct / t_total

    # --- Validate with TTA ---
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # TTA: original + horizontal flip
            out1 = torch.softmax(model(imgs), dim=1)
            out2 = torch.softmax(model(torch.flip(imgs, [3])), dim=1)
            avg = (out1 + out2) / 2
            v_loss += criterion(model(imgs), labels).item() * imgs.size(0)
            _, pred = avg.max(1)
            v_total += labels.size(0)
            v_correct += pred.eq(labels).sum().item()

    v_loss /= v_total
    v_acc = 100 * v_correct / v_total
    elapsed = time.time() - t0

    phase = "HEAD" if epoch <= FREEZE_EPOCHS else "FINE"
    mark = ""
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        mark = " ★ BEST"
        torch.save(model.state_dict(), "best_skin_model.pth")

    if v_loss < best_val_loss:
        best_val_loss = v_loss
        no_improve = 0
    else:
        no_improve += 1

    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)

    lr = optimizer.param_groups[0]['lr']
    print(f"\r  [{phase}] Epoch {epoch:2d}/{total_epochs} | "
          f"Train: {t_acc:5.1f}% ({t_loss:.4f}) | "
          f"Val: {v_acc:5.1f}% ({v_loss:.4f}) | "
          f"LR: {lr:.6f} | {elapsed:.0f}s{mark}")

    if epoch > FREEZE_EPOCHS and no_improve >= PATIENCE:
        print(f"\n  ⏹ Early stopping — no improvement for {PATIENCE} epochs")
        break

# Save class names
with open("class_names.json", "w") as f:
    json.dump(full_ds.classes, f, indent=2)

print(f"\n{'='*70}")
print(f"  ✅ TRAINING COMPLETE")
print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"  Model saved: best_skin_model.pth")
print(f"  Class names: class_names.json")
print(f"{'='*70}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5: Plot Training Curves                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["train_loss"], label="Train Loss", color="#0d9488", linewidth=2)
ax1.plot(history["val_loss"], label="Val Loss", color="#ef4444", linewidth=2)
ax1.set_title("Loss Curve", fontsize=14, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(history["train_acc"], label="Train Acc", color="#0d9488", linewidth=2)
ax2.plot(history["val_acc"], label="Val Acc", color="#3b82f6", linewidth=2)
ax2.axhline(y=best_val_acc, color="#22c55e", linestyle="--", alpha=0.5, label=f"Best: {best_val_acc:.1f}%")
ax2.set_title("Accuracy Curve", fontsize=14, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Saved: training_curves.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6: Download Trained Model                                         ║
# ║  Run this to download the model files to your PC                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# from google.colab import files
# files.download("best_skin_model.pth")
# files.download("class_names.json")
# files.download("training_curves.png")

# Then copy best_skin_model.pth to: e:\AI project\backend\
# And class_names.json to: e:\AI project\backend\
