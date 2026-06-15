import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

BASE = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026"

# GSL original 
GSL_META_CSV = os.path.join(BASE, "metadata.csv")
GSL_NPY_DIR  = os.path.join(BASE, "keypoints")

# Webcam (from custom data)
WEBCAM_META_CSV = os.path.join(BASE, "metadata_webcam.csv")
WEBCAM_NPY_DIR  = os.path.join(BASE, "keypoints_webcam")

# Output
OUTPUT_MODEL = os.path.join(BASE, "lstm", "gsl_lstm_merged.pt")

SEQ_LEN     = 40 # every gloss is captured as 40 frames
NUM_FEAT    = 126 # 21 keypoints extract by mediapipe x 3 (RGB) x 2 each for hand 
NUM_CLASSES = 10 # classes to classify

HIDDEN_SIZE = 128 
BATCH_SIZE  = 32
EPOCHS      = 200
LR          = 0.001
PATIENCE    = 15
LR_PATIENCE = 5
LR_FACTOR   = 0.5
SEED        = 42

IDX_TO_GLOSS = {
    0: 'ΓΕΙΑ',      1: 'ΕΥΧΑΡΙΣΤΩ', 2: 'ΠΑΡΑΚΑΛΩ',
    3: 'ΝΑΙ',       4: 'ΟΧΙ',       5: 'ΕΝΤΑΞΕΙ',
    6: 'ΘΕΛΩ',      7: 'ΜΠΟΡΩ',     8: 'ΧΡΕΙΑΖΟΜΑΙ',
    9: 'ΒΟΗΘΕΙΑ',
}

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# DATASET
class MergedDataset(Dataset):
    def __init__(self, samples):
        # samples: list of (np.array (40,126), int label)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.FloatTensor(seq), torch.LongTensor([label])[0]


class GSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=NUM_FEAT,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )
        self.fc1  = nn.Linear(HIDDEN_SIZE, 64)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        mask    = (x.abs().sum(dim=2) != 0)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        out = h_n.squeeze(0)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# TRAIN / EVAL
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(seqs)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for seqs, labels in loader:
        seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
        outputs = model(seqs)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * seqs.size(0)
        preds       = outputs.argmax(1)
        correct    += preds.eq(labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# LOAD DATA
def load_gsl_samples():
    print("\n[1] Φόρτωση GSL samples...")
    if not os.path.exists(GSL_META_CSV):
        print(f"    [ERROR] Δεν βρέθηκε: {GSL_META_CSV}")
        return [], [], []

    df      = pd.read_csv(GSL_META_CSV)
    train_s, val_s, test_s = [], [], []
    skipped = 0

    for _, row in df.iterrows():
        npy_path = os.path.join(GSL_NPY_DIR, row['npy_path'])
        if not os.path.exists(npy_path):
            skipped += 1
            continue
        seq   = np.load(npy_path).astype(np.float32)
        label = int(row['label'])
        split = row['split']
        if split == 'train':
            train_s.append((seq, label))
        elif split == 'val':
            val_s.append((seq, label))
        else:
            test_s.append((seq, label))

    print(f"    GSL train={len(train_s)} val={len(val_s)} test={len(test_s)}"
          f" (παραλείφθηκαν={skipped})")
    return train_s, val_s, test_s


def load_webcam_samples():
    print("\n[2] Φόρτωση Webcam samples...")
    if not os.path.exists(WEBCAM_META_CSV):
        print(f"    [WARN] Δεν βρέθηκε: {WEBCAM_META_CSV}")
        print(f"    Τρέξε πρώτα extract_keypoints_videos.py")
        return [], []

    df      = pd.read_csv(WEBCAM_META_CSV)
    samples = []
    skipped = 0

    for _, row in df.iterrows():
        npy_path = os.path.join(WEBCAM_NPY_DIR, row['npy_path'])
        if not os.path.exists(npy_path):
            skipped += 1
            continue
        # Παράλειψε samples χωρίς ανίχνευση χεριού
        if row.get('num_frames', 1) == 0:
            skipped += 1
            continue
        seq   = np.load(npy_path).astype(np.float32)
        label = int(row['label'])
        samples.append((seq, label))

    # Εμφάνιση ανά κλάση
    counts = Counter(label for _, label in samples)
    for idx, g in IDX_TO_GLOSS.items():
        print(f"    {g:15s}: {counts.get(idx, 0)} webcam samples")
    print(f"    Σύνολο webcam: {len(samples)} (παραλείφθηκαν={skipped})")

    # 80% train, 20% val
    np.random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    return samples[:split_idx], samples[split_idx:]


# PLOTS
def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'],   label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.close()


def plot_confusion(y_true, y_pred, out_path):
    cm     = confusion_matrix(y_true, y_pred)
    labels = [IDX_TO_GLOSS[i] for i in range(NUM_CLASSES)]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix — Merged Model (Test Set)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.close()


def plot_classification_report(y_true, y_pred, out_path):
    report = classification_report(
        y_true, y_pred,
        target_names=[IDX_TO_GLOSS[i] for i in range(NUM_CLASSES)],
        digits=4, output_dict=True
    )
    df = pd.DataFrame(report).transpose()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    table = ax.table(
        cellText=[[f"{v:.4f}" if isinstance(v, float) else str(v)
                   for v in row] for _, row in df.iterrows()],
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Classification Report — LSTM', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()
    
    
# MAIN
def main():
    # 1. Φόρτωση
    gsl_train, gsl_val, gsl_test     = load_gsl_samples()
    webcam_train, webcam_val          = load_webcam_samples()

    if not gsl_train:
        print("[ERROR] Δεν βρέθηκαν GSL training samples!")
        return

    # 2. Συνδυασμός
    print("\n[3] Συνδυασμός datasets...")
    train_samples = gsl_train + webcam_train
    val_samples   = gsl_val   + webcam_val
    test_samples  = gsl_test   # test μόνο GSL (αμερόληπτο)

    np.random.shuffle(train_samples)
    np.random.shuffle(val_samples)

    print(f"    Train: {len(gsl_train)} GSL + {len(webcam_train)} webcam"
          f" = {len(train_samples)}")
    print(f"    Val:   {len(gsl_val)} GSL + {len(webcam_val)} webcam"
          f" = {len(val_samples)}")
    print(f"    Test:  {len(test_samples)} GSL only")

    # Κατανομή ανά κλάση στο train
    counts = Counter(label for _, label in train_samples)
    print("\n    Κατανομή train ανά κλάση:")
    for idx, g in IDX_TO_GLOSS.items():
        print(f"      {g:15s}: {counts.get(idx, 0)}")

    # 3. DataLoaders
    train_ds = MergedDataset(train_samples)
    val_ds   = MergedDataset(val_samples)
    test_ds  = MergedDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # 4. Model
    model     = GSLModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE
    )
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Training
    print(f"\n[4] Εκπαίδευση ({EPOCHS} epochs, early stopping {PATIENCE})...")
    print("-" * 65)

    history        = defaultdict(list)
    best_val_loss  = float('inf')
    patience_count = 0
    best_state     = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc       = train_epoch(model, train_loader,
                                            criterion, optimizer)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        marker = ""
        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            best_state     = {k: v.clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
            marker         = " ← best"
        else:
            patience_count += 1

        print(f"Epoch {epoch:3d} | "
              f"Train {tr_loss:.4f}/{tr_acc:.3f} | "
              f"Val {vl_loss:.4f}/{vl_acc:.3f}{marker}")

        if patience_count >= PATIENCE:
            print(f"Early stopping στο epoch {epoch}")
            break

    # 6. Test
    print("\n" + "=" * 65)
    print("TEST EVALUATION (GSL only — αμερόληπτο)")
    print("=" * 65)
    model.load_state_dict(best_state)
    te_loss, te_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"Test Loss    : {te_loss:.4f}")
    print(f"Test Accuracy: {te_acc:.4f} ({te_acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(
        labels, preds,
        target_names=[IDX_TO_GLOSS[i] for i in range(NUM_CLASSES)],
        digits=4
    ))

    # plot_classification_report(labels, preds,
    # os.path.join(out_dir, "classification_report_merged.png"))
    
    # 7. Plots
    out_dir = os.path.join(BASE, "lstm")
    os.makedirs(out_dir, exist_ok=True)
    plot_history(history,
                 os.path.join(out_dir, "training_curves_merged.png"))
    plot_confusion(labels, preds,
                   os.path.join(out_dir, "confusion_matrix_merged.png"))
    plot_classification_report(labels, preds,
                 os.path.join(out_dir, "classification_report_merged.png"))
    # 8. Save
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'model_state_dict': best_state,
        'idx_to_gloss':     IDX_TO_GLOSS,
        'num_classes':      NUM_CLASSES,
        'seq_len':          SEQ_LEN,
        'num_feat':         NUM_FEAT,
    }, OUTPUT_MODEL)
    print(f"\n[SAVE] {OUTPUT_MODEL}")
    print(f"\nΕπόμενο βήμα:")
    print(f"  python live_inference.py --model {OUTPUT_MODEL}")

if __name__ == "__main__":
    main()