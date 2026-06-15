import cv2
import mediapipe as mp
import pandas as pd
import pickle

print("\n" + "="*50)
print("ΦΟΡΤΩΣΗ ΕΞΥΠΝΟΥ ΣΥΣΤΗΜΑΤΟΣ (PRO VERSION)...")
print("="*50)

# 1. Φορτώνουμε τον "εγκέφαλο" (Το MLP μοντέλο)
with open('mlp_neuralnet_model_relative.pkl', 'rb') as f:
    model = pickle.load(f)

# --- ΠΡΟΣΘΗΚΗ: Φορτώνουμε τον "μεταφραστή" γραμμάτων (Label Encoder) ---
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

columns = []
for i in range(21):
    columns.extend([f'x{i}', f'y{i}'])

display_map = {
    'A': 'A', 'B': 'B', 'Γ': 'Gamma', 'Δ': 'Delta', 'E': 'E', 'Z': 'Z',
    'H': 'H', 'Θ': 'Theta', 'I': 'I', 'K': 'K', 'Λ': 'Lambda', 'M': 'M',
    'N': 'N', 'Ξ': 'Xi', 'O': 'O', 'Π': 'Pi', 'P': 'P', 'Σ': 'Sigma',
    'T': 'T', 'Y': 'Y', 'Φ': 'Phi', 'X': 'X', 'Ψ': 'Psi', 'Ω': 'Omega'
}

# 2. Ξεκινάμε το MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- ΤΟ ΦΤΙΑΞΙΔΩΜΑ: Φανταχτερά χρώματα για τον σκελετό ---
# Αλλάζουμε το BGR (Blue, Green, Red) του OpenCV. 
# Προσοχή: Το OpenCV διαβάζει τα χρώματα ανάποδα! (B, G, R)
custom_dot_style = mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=5) # Κίτρινες κουκκίδες
custom_line_style = mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2) # Ματζέντα γραμμές

def main():
    cap = cv2.VideoCapture(0)
    
    # Διαβάζουμε το πλάτος της κάμερας για να φτιάξουμε το γραφικό UI
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if width == 0: width = 640

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Καθρέφτισμα της κάμερας 
        frame = cv2.flip(frame, 1)

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # Φτιάχνουμε τη μαύρη μπάρα (HUD) στο πάνω μέρος της οθόνης
        cv2.rectangle(frame, (0, 0), (width, 85), (20, 20, 20), -1)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Ζωγραφίζουμε τον σκελετό με τα φανταχτερά στυλ
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                custom_dot_style, custom_line_style
            )
            
            # Εξαγωγή σχετικών συντεταγμένων
            row = []
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            for lm in hand_landmarks.landmark:
                row.extend([lm.x - wrist_x, lm.y - wrist_y])
            
            X_live = pd.DataFrame([row], columns=columns)
            
            # --- ΠΡΟΣΘΗΚΗ: Πρόβλεψη και Μετάφραση του αποτελέσματος ---
            prediction_num = model.predict(X_live)[0] # Το μοντέλο εξάγει αριθμό (π.χ. 0)
            prediction = le.inverse_transform([prediction_num])[0] # Μετατροπή αριθμού σε γράμμα (π.χ. 'A')
            
            probabilities = model.predict_proba(X_live)[0]
            confidence = max(probabilities) * 100 # Το μετατρέπουμε σε ποσοστό
            
            display_text = display_map.get(prediction, prediction)
            
            # Επιλογή χρώματος ανάλογα με τη σιγουριά (Πράσινο=Καλό, Κίτρινο=Μέτριο, Κόκκινο=Κακό)
            if confidence > 80:
                conf_color = (0, 255, 0)
            elif confidence > 50:
                conf_color = (0, 255, 255)
            else:
                conf_color = (0, 0, 255)
            
            # Τυπώνουμε τα αποτελέσματα πάνω στη μαύρη μπάρα
            cv2.putText(frame, f"AI VISION | Letter: {display_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                        
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)

        else:
            # Τι δείχνει όταν δεν βρίσκει χέρι
            cv2.putText(frame, "AI VISION | Waiting for hand...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        cv2.imshow("Live Sign Language AI - PRO VERSION", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()