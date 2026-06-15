import os
import math
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
OUT_DIR = os.path.join(BASE, "Transformer")

GSL_META_CSV    = os.path.join(BASE, "metadata.csv")
GSL_NPY_DIR     = os.path.join(BASE, "keypoints")
WEBCAM_META_CSV = os.path.join(BASE, "metadata_webcam.csv")
WEBCAM_NPY_DIR  = os.path.join(BASE, "keypoints_webcam")
OUTPUT_MODEL    = os.path.join(OUT_DIR, "gsl_transformer.pt")

SEQ_LEN     = 40
NUM_FEAT    = 126
NUM_CLASSES = 10

# Transformer hyperparameters
D_MODEL     = 128   # διάσταση κάθε token μετά το projection
N_HEADS     = 4     # αριθμός attention heads (D_MODEL % N_HEADS == 0)
N_LAYERS    = 2     # αριθμός Transformer Encoder layers
DIM_FF      = 256   # διάσταση feed-forward layer
DROPOUT     = 0.1   # dropout για regularization

# Training
BATCH_SIZE  = 32
EPOCHS      = 200
LR          = 0.0005
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
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.FloatTensor(seq), torch.LongTensor([label])[0]


# POSITIONAL ENCODING (its the third thing in embedding layer) adds positional information in each token
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Υπολογισμός positional encoding matrix
        pe = torch.zeros(max_len, d_model)               # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )                                                  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)      # even indices
        pe[:, 1::2] = torch.cos(position * div_term)      # odd indices
        pe = pe.unsqueeze(0)                               # (1, max_len, d_model)
        self.register_buffer('pe', pe)                     # δεν εκπαιδεύεται

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# TRANSFORMER MODEL
class GSLTransformer(nn.Module):
    """
    Transformer Encoder για classification νοημάτων GSL.

    Pipeline:
        1. Linear projection: 126 → d_model (128)
        2. Positional Encoding: προσθέτει θέση κάθε frame
        3. Padding mask: αγνοεί all-zero frames (padding)
        4. Transformer Encoder: N layers self-attention + FFN
        5. Mean pooling: 40 tokens → 1 vector (αγνοώντας padding)
        6. Classifier: d_model → 64 → 10 classes
    """
    def __init__(self):
        super().__init__()

        # 1. Linear projection: 126 features → d_model
        self.input_proj = nn.Linear(NUM_FEAT, D_MODEL) # D_Model = 128

        # 2. Positional Encoding
        self.pos_enc = PositionalEncoding(D_MODEL, max_len=SEQ_LEN + 10,
                                          dropout=DROPOUT)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS, # 4 
            dim_feedforward=DIM_FF, # 256 
            dropout=DROPOUT,
            batch_first=True,    # (B, T, d_model) — ίδιο format με LSTM
            norm_first=True,     # Pre-LN: πιο σταθερή εκπαίδευση
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=N_LAYERS,
            norm=nn.LayerNorm(D_MODEL),
        )

        # 4. Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES),
        )

    def _make_padding_mask(self, x):
        """
        Φτιάχνει boolean mask για padding frames.
        True = αγνόησε αυτό το frame (all-zero = padding).
        Shape: (B, T)
        """
        return (x.abs().sum(dim=2) == 0)  # (B, T)

    def forward(self, x):
        # x: (B, 40, 126)
        pad_mask = self._make_padding_mask(x)   # (B, 40) — True για padding

        # 1. Linear projection
        out = self.input_proj(x)                # (B, 40, 128)

        # 2. Positional Encoding
        out = self.pos_enc(out)                 # (B, 40, 128)

        # 3. Transformer Encoder (παραλληλα σε όλα τα frames)
        out = self.transformer(
            out,
            src_key_padding_mask=pad_mask       # αγνοεί padding frames
        )                                       # (B, 40, 128)

        # 4. Mean pooling (αγνοώντας padding)
        # Αντεστρέφουμε το mask: True = valid frame
        valid_mask = (~pad_mask).float().unsqueeze(-1)  # (B, 40, 1)
        sum_out    = (out * valid_mask).sum(dim=1)       # (B, 128)
        count      = valid_mask.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled     = sum_out / count                     # (B, 128)

        # 5. Classifier
        return self.classifier(pooled)                   # (B, 10)


# ============================================================
# TRAIN / EVAL
# ============================================================
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
    plt.title('Classification Report — Transformer', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

# ============================================================
# LOAD DATA (ίδιο με LSTM/BiLSTM)
# ============================================================
def load_gsl_samples():
    print("\n[1] Φόρτωση GSL samples...")
    if not os.path.exists(GSL_META_CSV):
        print(f"    [ERROR] Δεν βρέθηκε: {GSL_META_CSV}")
        return [], [], []

    df = pd.read_csv(GSL_META_CSV)
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
        return [], []

    df = pd.read_csv(WEBCAM_META_CSV)
    samples = []
    skipped = 0

    for _, row in df.iterrows():
        npy_path = os.path.join(WEBCAM_NPY_DIR, row['npy_path'])
        if not os.path.exists(npy_path):
            skipped += 1
            continue
        if row.get('num_frames', 1) == 0:
            skipped += 1
            continue
        seq   = np.load(npy_path).astype(np.float32)
        label = int(row['label'])
        samples.append((seq, label))

    counts = Counter(label for _, label in samples)
    for idx, g in IDX_TO_GLOSS.items():
        print(f"    {g:15s}: {counts.get(idx, 0)} webcam samples")
    print(f"    Σύνολο webcam: {len(samples)} (παραλείφθηκαν={skipped})")

    np.random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    return samples[:split_idx], samples[split_idx:]


# ============================================================
# PLOTS
# ============================================================
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
    plt.title('Confusion Matrix — Transformer (Test Set)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Φόρτωση δεδομένων
    gsl_train, gsl_val, gsl_test = load_gsl_samples()
    webcam_train, webcam_val     = load_webcam_samples()

    if not gsl_train:
        print("[ERROR] Δεν βρέθηκαν GSL training samples!")
        return

    # 2. Συνδυασμός
    print("\n[3] Συνδυασμός datasets...")
    train_samples = gsl_train + webcam_train
    val_samples   = gsl_val   + webcam_val
    test_samples  = gsl_test

    np.random.shuffle(train_samples)
    np.random.shuffle(val_samples)

    print(f"    Train: {len(gsl_train)} GSL + {len(webcam_train)} webcam"
          f" = {len(train_samples)}")
    print(f"    Val:   {len(gsl_val)} GSL + {len(webcam_val)} webcam"
          f" = {len(val_samples)}")
    print(f"    Test:  {len(test_samples)} GSL only")

    counts = Counter(label for _, label in train_samples)
    print("\n    Κατανομή train ανά κλάση:")
    for idx, g in IDX_TO_GLOSS.items():
        print(f"      {g:15s}: {counts.get(idx, 0)}")

    # 3. DataLoaders
    train_loader = DataLoader(MergedDataset(train_samples),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(MergedDataset(val_samples),
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(MergedDataset(test_samples),
                              batch_size=BATCH_SIZE)

    # 4. Model
    model = GSLTransformer().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total_params:,}")
    print(f"Architecture:")
    print(f"  d_model={D_MODEL}, n_heads={N_HEADS}, "
          f"n_layers={N_LAYERS}, dim_ff={DIM_FF}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE
    )

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

    report = classification_report(
        labels, preds,
        target_names=[IDX_TO_GLOSS[i] for i in range(NUM_CLASSES)],
        digits=4
    )
    print("\nClassification Report:")
    print(report)

    # 7. Plots + report
    plot_history(history,
                 os.path.join(OUT_DIR, "training_curves_transformer.png"))
    plot_confusion(labels, preds,
                   os.path.join(OUT_DIR, "confusion_matrix_transformer.png"))

    report_path = os.path.join(OUT_DIR, "classification_report_transformer.txt")
    
    plot_classification_report(labels, preds,
    os.path.join(OUT_DIR, "classification_report_transformer.png"))
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Loss    : {te_loss:.4f}\n")
        f.write(f"Test Accuracy: {te_acc:.4f} ({te_acc*100:.2f}%)\n\n")
        f.write(f"Architecture:\n")
        f.write(f"  d_model={D_MODEL}, n_heads={N_HEADS}, "
                f"n_layers={N_LAYERS}, dim_ff={DIM_FF}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved: {report_path}")

    # 8. Save model
    torch.save({
        'model_state_dict': best_state,
        'idx_to_gloss':     IDX_TO_GLOSS,
        'num_classes':      NUM_CLASSES,
        'seq_len':          SEQ_LEN,
        'num_feat':         NUM_FEAT,
        'd_model':          D_MODEL,
        'n_heads':          N_HEADS,
        'n_layers':         N_LAYERS,
        'dim_ff':           DIM_FF,
        'dropout':          DROPOUT,
    }, OUTPUT_MODEL)
    print(f"\n[SAVE] {OUTPUT_MODEL}")
    print(f"\nΕπόμενο βήμα:")
    print(f"  python live_inference_transformer.py --model {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()