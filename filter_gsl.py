"""
Φιλτράρει το merged_isolated.csv και κρατά μόνο τα 10 επιλεγμένα glosses.
Αποθηκεύει νέο CSV: gsl_10class.csv
"""

ROOT = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/GSL_isolated/Greek_isolated/GSL_isol"

SELECTED = {
    'ΓΕΙΑ',
    'ΕΥΧΑΡΙΣΤΩ',
    'ΠΑΡΑΚΑΛΩ',
    'ΝΑΙ',
    'ΟΧΙ',
    'ΕΝΤΑΞΕΙ',
    'ΘΕΛΩ',
    'ΜΠΟΡΩ',
    'ΧΡΕΙΑΖΟΜΑΙ',
    'ΒΟΗΘΕΙΑ',
}

import os
from pathlib import Path
from collections import Counter

csv_in  = Path(ROOT) / "merged_isolated.csv"
csv_out = Path(ROOT) / "gsl_10class.csv"

kept    = []
skipped = 0

with open(csv_in, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('|')
        if len(parts) != 2:
            continue
        folder = parts[0].strip()
        gloss  = parts[1].strip()

        if gloss not in SELECTED:
            skipped += 1
            continue

        # Έλεγξε αν υπάρχει ο φάκελος
        full_path = Path(ROOT) / folder
        if not full_path.exists():
            print(f"  MISSING: {folder}")
            skipped += 1
            continue

        kept.append(f"{folder} | {gloss}")

with open(csv_out, 'w', encoding='utf-8') as f:
    for line in kept:
        f.write(line + '\n')

counts = Counter(line.split('|')[1].strip() for line in kept)
print(f"Αποθηκεύτηκε: {csv_out}")
print(f"Σύνολο: {len(kept)} | Παραλείφθηκαν: {skipped}")
print("\nΑνά κλάση:")
for g in sorted(SELECTED):
    print(f"  {g:15s}: {counts.get(g, 0):5d}")