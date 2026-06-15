# 🤝 Αναγνώριση Ελληνικής Νοηματικής Γλώσσας σε Πραγματικό Χρόνο

Ένα καινοτόμο σύστημα αναγνώρισης ελληνικών λέξεων και φράσεων νοηματικής γλώσσας σε πραγματικό χρόνο, με ζωντανή μετάφραση μέσω του Cloudflare.

---

## ✨ Χαρακτηριστικά

✅ **Αναγνώριση σε Πραγματικό Χρόνο** - Ανίχνευση νοημάτων καθώς γίνονται  
✅ **Ελληνικό Αλφάβητο** - Πλήρης υποστήριξη ελληνικών γραμμάτων και νοημάτων  
✅ **Custom Dataset** - Εκπαίδευση σε αυθεντικά δεδομένα ελληνικής νοηματικής γλώσσας  
✅ **Ζωντανή Μετάφραση** - Άμεση μετάφραση νοημάτων σε κείμενο  
✅ **Web Interface** - Εύχρηστη διεπαφή για όλους  
✅ **Cloudflare Integration** - Ασφαλής και γρήγορη διανομή μέσω Cloudflare Tunnel  

---

## 🛠️ Τεχνολογίες

| Τεχνολογία | Περιγραφή |
|-----------|-----------|
| **MediaPipe** | Ανίχνευση σκελετού χεριών και σώματος |
| **TensorFlow Lite** | Μοντέλο αναγνώρισης κίνησης (SPOTER) |
| **FastAPI** | Backend server |
| **OpenCV** | Επεξεργασία video σε πραγματικό χρόνο |
| **Cloudflare Tunnel** | Δημόσια πρόσβαση και ασφάλεια |
| **Python** | Κύρια γλώσσα ανάπτυξης |

---

## 🚀 Έναρξη

### Προαπαιτούμενα
- Python 3.10+
- Webcam ή κάμερα video
- Git

### Εγκατάσταση

1. **Κλωνοποίηση του repository:**
```bash
git clone https://github.com/CharalamposZisis/Greek-Sign-Language.git
cd Greek-Sign-Language
```

2. **Δημιουργία virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ή
venv\Scripts\activate  # Windows
```

3. **Εγκατάσταση απαιτούμενων βιβλιοθηκών:**
```bash
pip install -r requirements.txt
```

4. **Εκκίνηση του server:**
```bash
python server.py
```

---

## 📱 Χρήση

### Τοπικά (Local)
Ανοίξτε το browser και πηγαίνετε στο:
```
http://localhost:8000
```

### Δημόσιο Link μέσω Cloudflare
Για να κοινοποιήσετε το project με άλλους:

```bash
cloudflared tunnel --url http://localhost:8000
```

Θα λάβετε ένα δημόσιο URL όπως:
```
https://subsidiary-outstanding-flush-starts.trycloudflare.com
```

---

## 📊 Custom Dataset

Το project χρησιμοποιεί ένα **custom dataset** με:
- ✔️ Ελληνικά νοήματα και φράσεις
- ✔️ Πολλαπλές κάμερες και γωνίες
- ✔️ Διάφοροι χρήστες
- ✔️ Ποικιλία συνθηκών φωτισμού

### Δομή Dataset
```
data/
├── letter_a/
├── letter_b/
├── ...
├── phrase_hello/
├── phrase_thank_you/
└── ...
```

---

## 🎯 Πώς Λειτουργεί

```
1. Κάμερα Ενεργή
    ↓
2. MediaPipe Ανίχνευση (Χέρια, Σώμα)
    ↓
3. SPOTER Model (TensorFlow Lite)
    ↓
4. Αναγνώριση Νοήματος
    ↓
5. Ζωντανή Μετάφραση σε Κείμενο
    ↓
6. Εμφάνιση στην Οθόνη
```

---

## 🌐 Cloudflare Integration

Το σύστημα χρησιμοποιεί **Cloudflare Tunnel** για:
- 🔒 Ασφαλή πρόσβαση
- ⚡ Γρήγορη διανομή
- 🌍 Παγκόσμια Προσβασιμότητα
- 📍 Δυναμικά URLs

### Κανόνες Ασφάλειας
- Το tunnel δεν χρειάζεται firewall configuration
- Αυτόματη δρομολόγηση μέσω Cloudflare
- Προστασία από DDoS

---

## 📝 Ελληνικές Κατηγορίες

### Γράμματα Αλφάβητου
Α, Β, Γ, Δ, Ε, Ζ, Η, Θ, Ι, Κ, Λ, Μ, Ν, Ξ, Ο, Π, Ρ, Σ, Τ, Υ, Φ, Χ, Ψ, Ω

### Συνηθισμένες Φράσεις
- "Καλημέρα" (Kalispéra)
- "Ευχαριστώ" (Efharistó)
- "Παρακαλώ" (Parakaló)
- "Ναι" / "Όχι" (Ne / Óhi)
- Και πολλά άλλα...

---

## 📁 Δομή Project

```
Greek-Sign-Language/
├── server.py              # FastAPI server
├── requirements.txt       # Απαιτούμενες βιβλιοθήκες
├── models/                # Προεκπαιδευμένα μοντέλα
├── data/                  # Custom dataset
├── static/                # Frontend files (HTML, CSS, JS)
├── templates/             # HTML templates
└── README.md              # Αυτό το αρχείο
```

---

## 🎓 Εκπαίδευση Μοντέλου

Για να εκπαιδεύσετε το μοντέλο με νέα δεδομένα:

```bash
python train.py --dataset data/ --epochs 100
```

---

## 🤝 Συνεργάτες

- **Χαράλαμπος Ζήσης** - Developer & ML Engineer
- **tsamis** - Collaborator (invited)

---

## 📜 Άδεια

Αυτό το project είναι ανοικτού κώδικα και διατίθεται υπό την άδεια MIT.

---

## 🐛 Αναφορά Προβλημάτων

Εάν συναντήσετε κάποιο πρόβλημα:

1. Ανοίξτε ένα [Issue](https://github.com/CharalamposZisis/Greek-Sign-Language/issues)
2. Περιγράψτε το πρόβλημα με λεπτομέρεια
3. Συμπεριλάβετε logs ή screenshots

---

## 💡 Βελτιώσεις στο Μέλλον

- 🔄 Υποστήριξη περισσότερων ελληνικών νοημάτων
- 🎯 Βελτίωση ακρίβειας μοντέλου
- 📱 Mobile app
- 🔊 Προσθήκη φωνής
- 🌐 Πολυγλωσσική υποστήριξη

---

## 📞 Επικοινωνία

Για ερωτήσεις ή προτάσεις:
- GitHub Issues: [Greek-Sign-Language Issues](https://github.com/CharalamposZisis/Greek-Sign-Language/issues)
- Repository: [Greek-Sign-Language](https://github.com/CharalamposZisis/Greek-Sign-Language)

---

## 🙏 Ευχαριστίες

Ευχαριστούμε τους contributors και τη κοινότητα που στηρίζει αυτό το project!

---

**Τελευταία ενημέρωση:** Ιούνιος 2026  
**Κατάσταση:** ✅ Ενεργή Ανάπτυξη
