import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

print("\n" + "="*75)
print(" MASTER SCRIPT: ΣΥΓΚΡΙΣΗ ΜΟΝΤΕΛΩΝ & ΚΑΜΠΥΛΕΣ ΕΚΠΑΙΔΕΥΣΗΣ (ΓΡΑΜΜΑΤΑ)")
print("="*75)

# Δίνουμε το ΑΠΟΛΥΤΟ PATH για να μην μπερδεύεται ποτέ το VS Code
BASE_DIR = r"C:\Users\xrist\Desktop\Masters\Β Εξάμηνο\Βαθιά Μάθηση και Τεχνητή Νοημοσύνη\static_point_rf"
CSV_FILE = "Final_final_merged_dataset_relative.csv" 
CSV_PATH = os.path.join(BASE_DIR, CSV_FILE)

# 1. Φόρτωση Δεδομένων
print(f"[1/5] Φόρτωση δεδομένων από: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
except FileNotFoundError:
    print("\nΣΦΑΛΜΑ: Το αρχείο CSV δεν βρέθηκε! Βεβαιώσου για το όνομα.")
    exit()

y_strings = df['label']
X = df.drop('label', axis=1)

# 2. Μετατροπή των Γραμμάτων σε Αριθμούς (Label Encoding) για το Early Stopping
print("[2/5] Κωδικοποίηση γραμμάτων και δημιουργία Label Encoder...")
le = LabelEncoder()
y_encoded = le.fit_transform(y_strings)

# Αποθήκευση του "Μεταφραστή" για το Live script
encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")
with open(encoder_path, 'wb') as f:
    pickle.dump(le, f)

# Χωρισμός με τα κωδικοποιημένα labels
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
num_test_samples = len(X_test)

# 3. Ορισμός των 4 Μοντέλων (το MLP έχει Early Stopping)
models = {
    "Random_Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "MLP_NeuralNet": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True, validation_fraction=0.1, random_state=42)
}

accuracies = {}
inference_times = {}

# 4. Εκπαίδευση, Χρονομέτρηση και Αποθήκευση
print(f"\n[3/5] Ξεκινάει η εκπαίδευση και η χρονομέτρηση (Test Set: {num_test_samples} δείγματα)...\n")

for name, model in models.items():
    print(f"[*] Αξιολόγηση: {name} ...")
    
    t0_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0_train
    
    t0_inf = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_time_total = time.perf_counter() - t0_inf
    
    inf_time_per_sample_ms = (inf_time_total / num_test_samples) * 1000
    inference_times[name] = inf_time_per_sample_ms
    
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc * 100
    
    print(f"    --> Ακρίβεια:   {accuracies[name]:.2f}%")
    print(f"    --> Χρόνος Train: {train_time:.2f} sec")
    print(f"    --> Χρόνος Live:  {inf_time_per_sample_ms:.4f} ms ανά frame")
    
    if name == "MLP_NeuralNet":
        print(f"    --> Early Stop:   Στις {model.n_iter_} εποχές")
    print()
    
    # Αποθήκευση του μοντέλου
    filename = os.path.join(BASE_DIR, f"{name.lower()}_model_relative.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# 5. Γραφήματα Σύγκρισης για την Αναφορά
print("[4/5] Δημιουργία των 3 γραφημάτων...")
colors = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0']

# Γράφημα 1: Ακρίβεια
plt.figure(figsize=(10, 5))
bars1 = plt.bar(accuracies.keys(), accuracies.values(), color=colors)
plt.title('Σύγκριση Ακρίβειας Μοντέλων (Υψηλότερο = Καλύτερο)')
plt.ylabel('Ακρίβεια (%)')
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.2f}%", ha='center', fontweight='bold')
plt.savefig(os.path.join(BASE_DIR, 'model_comparison_accuracy.png'), dpi=150)
plt.close()

# Γράφημα 2: Ταχύτητα
plt.figure(figsize=(10, 5))
bars2 = plt.bar(inference_times.keys(), inference_times.values(), color=['#FF5722', '#00BCD4', '#8BC34A', '#E91E63'])
plt.title('Χρόνος Απόκρισης (Inference Time) ανά Frame (Χαμηλότερο = Ταχύτερο)')
plt.ylabel('Milliseconds (ms)')
plt.ylim(0, max(inference_times.values()) * 1.3)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(inference_times.values()) * 0.05), f"{yval:.4f} ms", ha='center', fontweight='bold')
plt.savefig(os.path.join(BASE_DIR, 'model_comparison_time.png'), dpi=150)
plt.close()

# Γράφημα 3: Καμπύλη Εκπαίδευσης MLP
mlp_model = models["MLP_NeuralNet"]
plt.figure(figsize=(9, 5))
plt.plot(mlp_model.validation_scores_, label='Ακρίβεια Επαλήθευσης (Validation Score)', color='#E91E63', linewidth=2.5)
plt.title('Καμπύλη Σύγκλισης Νευρωνικού Δικτύου (MLP με Early Stopping)', fontweight='bold')
plt.xlabel('Εποχές (Epochs)')
plt.ylabel('Ακρίβεια')
plt.ylim(0.8, 1.0) 
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.axvline(x=len(mlp_model.validation_scores_)-1, color='gray', linestyle='--')
plt.text(len(mlp_model.validation_scores_)-2, 0.85, f' Early Stop\n (Εποχή {mlp_model.n_iter_})', ha='right', color='gray')
plt.savefig(os.path.join(BASE_DIR, 'mlp_learning_curve.png'), dpi=150)
plt.close()

print("[5/5] Τα γραφήματα αποθηκεύτηκαν επιτυχώς!\n")
print("="*75)
print("ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!")
print("Εικόνες: 'model_comparison_accuracy.png', 'model_comparison_time.png', 'mlp_learning_curve.png'")
print("="*75 + "\n")