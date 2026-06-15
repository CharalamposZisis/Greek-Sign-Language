import cv2
import mediapipe as mp
import csv
import os
import time

# ============================================================
# ΡΥΘΜΙΣΕΙΣ ΑΡΧΕΙΟΥ CSV & ΜΕΤΑΦΡΑΣΗ ΠΛΗΚΤΡΟΛΟΓΙΟΥ
# ============================================================
csv_file = 'greek_hand_dataset.csv'

# Αντιστοιχία: Αγγλικό Πλήκτρο -> Ελληνικό Γράμμα
KEY_MAP = {
    ord('a'): 'A', ord('b'): 'B', ord('g'): 'Γ', ord('d'): 'Δ',
    ord('e'): 'E', ord('z'): 'Z', ord('h'): 'H', ord('u'): 'Θ',
    ord('i'): 'I', ord('k'): 'K', ord('l'): 'Λ', ord('m'): 'M',
    ord('n'): 'N', ord('j'): 'Ξ', ord('o'): 'O', ord('p'): 'Π',
    ord('r'): 'P', ord('s'): 'Σ', ord('t'): 'T', ord('y'): 'Y',
    ord('f'): 'Φ', ord('x'): 'X', ord('c'): 'Ψ', ord('v'): 'Ω'
}

# Δημιουργία αρχείου και στηλών αν δεν υπάρχει
if not os.path.isfile(csv_file):
    columns = ['label']
    for i in range(21):
        columns.extend([f'x{i}', f'y{i}'])
        
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

# Αρχικοποίηση μετρητών (διαβάζει πόσα έχουμε ήδη στο CSV)
counters = {val: 0 for val in KEY_MAP.values()}
if os.path.isfile(csv_file):
    with open(csv_file, mode='r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Προσπερνάμε την επικεφαλίδα
        for row in reader:
            if row and row[0] in counters:
                counters[row[0]] += 1

# ============================================================
# SETUP MEDIAPIPE
# ============================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)
    
    print("\n" + "="*50)
    print("ΣΥΛΛΟΓΗ ΔΕΔΟΜΕΝΩΝ - ΕΛΛΗΝΙΚΗ ΝΟΗΜΑΤΙΚΗ (24 ΓΡΑΜΜΑΤΑ)")
    print("="*50)
    print("ΠΑΤΑ ΤΟ ΑΝΤΙΣΤΟΙΧΟ ΑΓΓΛΙΚΟ ΓΡΑΜΜΑ ΓΙΑ ΕΝΑΡΞΗ:")
    print("a:A, b:B, g:Γ, d:Δ, e:E, z:Z, h:H, u:Θ")
    print("i:I, k:K, l:Λ, m:M, n:N, j:Ξ, o:O, p:Π")
    print("r:P, s:Σ, t:T, y:Y, f:Φ, x:X, c:Ψ, v:Ω")
    print("-" * 50)
    print("[SPACE] : ΠΑΥΣΗ καταγραφής.")
    print("[Q]     : ΕΞΟΔΟΣ από το πρόγραμμα.")
    print("="*50 + "\n")
    
    recording = False
    current_label = ""
    last_save_time = 0
    save_interval = 0.1  # Αποθήκευση 10 φορές το δευτερόλεπτο

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        key = cv2.waitKey(1) & 0xFF

        # Ελέγχουμε αν πατήθηκε γράμμα για καταγραφή
        if key in KEY_MAP:
            current_label = KEY_MAP[key]
            recording = True
            print(f"--- ΓΡΑΦΟΥΜΕ ΤΟ ΓΡΑΜΜΑ: '{current_label}' | Συνολικά: {counters[current_label]} ---")
        
        # Παύση με το SPACE (Κενό, ascii: 32)
        elif key == 32: 
            if recording:
                recording = False
                print("--- Η ΚΑΤΑΓΡΑΦΗ ΣΤΑΜΑΤΗΣΕ ---")
                current_label = ""
                
        # Έξοδος με Q
        elif key == ord('q'):
            break

        # Λογική Καταγραφής Σημείων
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            current_time = time.time()
            if recording and (current_time - last_save_time > save_interval):
                row = [current_label]
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y])
                
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                counters[current_label] += 1
                last_save_time = current_time

        # --- Ενδείξεις στην οθόνη ---
        if recording:
            # Δείχνει το γράμμα που γράφουμε και τον μετρητή του
            cv2.rectangle(frame, (0, 0), (350, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"REC: {current_label} ({counters[current_label]})", (10, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, "Press SPACE to Pause", (10, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.rectangle(frame, (0, 0), (450, 50), (0, 0, 0), -1)
            cv2.putText(frame, "PAUSED - Press a key to start", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Greek Sign Language - Data Collection", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()