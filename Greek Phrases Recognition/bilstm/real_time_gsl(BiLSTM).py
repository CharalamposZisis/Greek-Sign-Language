"""
GSL Live Inference - MediaPipe Tasks API + PIL Greek Text
=========================================================
Πλήκτρα:
    SPACE  : manual trigger (εγγράφει τα επόμενα 40 frames)
    A      : auto mode on/off (motion detection)
    C      : καθαρισμός πρότασης
    S      : αποθήκευση πρότασης
    Q/ESC  : έξοδος
"""

import argparse
import collections
import time
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import ImageFont, ImageDraw, Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ============================================================
# ΡΥΘΜΙΣΕΙΣ
# ============================================================
DEFAULT_MODEL = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/bilstm/gsl_Bilstm_merged.pt"
MODEL_PATH    = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/hand_landmarker.task"
DEFAULT_CAM   = 0
CONF_THRESHOLD = 0.60
SEQ_LEN  = 40
NUM_FEAT = 126

MOTION_THRESHOLD  = 0.015
MOTION_WINDOW     = 5
STILLNESS_FRAMES  = 12
MIN_ACTIVE_FRAMES = 10
DUPLICATE_WINDOW  = 1.5

COLOR_GREEN  = (0, 220, 0)
COLOR_RED    = (0, 0, 220)
COLOR_BLUE   = (255, 160, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (160, 160, 160)
COLOR_YELLOW = (0, 220, 220)

FONT_PATH      = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ============================================================
# GREEK TEXT με PIL
# ============================================================
def put_greek_text(frame, text, pos, font_size=20, color=(255, 255, 255)):
    """Γράφει ελληνικό/unicode κείμενο — color σε BGR."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    rgb = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def put_greek_text_bold(frame, text, pos, font_size=22, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH_BOLD, font_size)
    except:
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except:
            font = ImageFont.load_default()
    rgb = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# MODEL
# ============================================================
class GSLModel(nn.Module):
    def __init__(self, num_classes, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=NUM_FEAT,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,   # ← πρόσθεσε αυτό
        )
        self.fc1  = nn.Linear(2 * hidden_size, 64)  # ← 256 αντί 128
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(64, num_classes)

    def forward(self, x):
        mask    = (x.abs().sum(dim=2) != 0)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        out = torch.cat([h_n[0], h_n[1]], dim=1)  # ← αντί h_n.squeeze(0)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# ============================================================
# KEYPOINT EXTRACTION — ίδιο με extract_keypoints.py
# ============================================================
def extract_frame_keypoints(frame_bgr, detector):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result    = detector.detect(mp_image)

    left_kp  = np.zeros(63, dtype=np.float32)
    right_kp = np.zeros(63, dtype=np.float32)

    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks,
                                         result.handedness):
            label = handedness[0].category_name
            kp    = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks],
                dtype=np.float32
            ).flatten()
            if label == 'Left':
                right_kp = kp
            else:
                left_kp  = kp

    return np.concatenate([left_kp, right_kp]), result


# ============================================================
# MOTION DETECTION
# ============================================================
def compute_motion(prev_kp, curr_kp):
    mask = (prev_kp != 0) & (curr_kp != 0)
    if mask.sum() == 0:
        return 0.0
    return float(np.abs(curr_kp[mask] - prev_kp[mask]).mean())


class GestureSegmenter:
    IDLE   = "IDLE"
    ACTIVE = "ACTIVE"

    def __init__(self):
        self.state          = self.IDLE
        self.buffer         = []
        self.still_count    = 0
        self.motion_history = collections.deque(maxlen=MOTION_WINDOW)
        self.prev_kp        = None

    def update(self, kp):
        segment = None
        motion  = 0.0
        if self.prev_kp is not None:
            motion = compute_motion(self.prev_kp, kp)
        self.motion_history.append(motion)
        self.prev_kp = kp.copy()

        smooth_motion = np.mean(self.motion_history)
        is_moving     = smooth_motion > MOTION_THRESHOLD
        has_hands     = kp.sum() != 0

        if self.state == self.IDLE:
            if is_moving and has_hands:
                self.state       = self.ACTIVE
                self.buffer      = [kp.copy()]
                self.still_count = 0
        elif self.state == self.ACTIVE:
            self.buffer.append(kp.copy())
            if not is_moving or not has_hands:
                self.still_count += 1
            else:
                self.still_count = 0
            if (self.still_count >= STILLNESS_FRAMES or
                    len(self.buffer) >= SEQ_LEN):
                if len(self.buffer) >= MIN_ACTIVE_FRAMES:
                    segment = self._build_sequence()
                self._reset()

        return segment, smooth_motion, self.state

    def _build_sequence(self):
        arr = np.array(self.buffer, dtype=np.float32)
        T   = len(arr)
        if T >= SEQ_LEN:
            return arr[:SEQ_LEN]
        pad = np.zeros((SEQ_LEN - T, NUM_FEAT), dtype=np.float32)
        return np.vstack([arr, pad])

    def _reset(self):
        self.state       = self.IDLE
        self.buffer      = []
        self.still_count = 0


# ============================================================
# MANUAL RECORDING
# ============================================================
class ManualRecorder:
    def __init__(self):
        self.recording = False
        self.buffer    = []

    def start(self):
        self.recording = True
        self.buffer    = []

    def update(self, kp):
        if not self.recording:
            return None
        self.buffer.append(kp.copy())
        if len(self.buffer) >= SEQ_LEN:
            seq = np.array(self.buffer, dtype=np.float32)
            self.recording = False
            self.buffer    = []
            return seq
        return None

    @property
    def progress(self):
        return len(self.buffer)


# ============================================================
# INFERENCE
# ============================================================
@torch.no_grad()
def run_inference(model, sequence, device):
    x      = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    logits = model(x)
    probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])
    return idx, conf, probs


# ============================================================
# DISPLAY
# ============================================================
def draw_landmarks_on_frame(frame, result):
    if not result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    for hand_landmarks in result.hand_landmarks:
        pts = [(int((1 - lm.x) * w), int(lm.y * h))
               for lm in hand_landmarks]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 100, 255), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 220, 0), -1)


def draw_ui(frame, state_auto, auto_mode, manual_rec,
            smooth_motion, last_word, last_conf,
            sentence_words, last_probs, idx_to_gloss, fps):
    h, w = frame.shape[:2]

    # ── Πάνω μπάρα ──────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 105), (30, 30, 30), -1)

    # Πρόταση (ελληνικά με PIL)
    text = " ".join(sentence_words) if sentence_words else "..."
    put_greek_text_bold(frame, text, (10, 5),
                        font_size=26, color=COLOR_GREEN)

    # Mode
    if auto_mode:
        put_greek_text(frame, f"Mode: AUTO  [A]=toggle",
                       (10, 42), font_size=18, color=COLOR_YELLOW)
    else:
        put_greek_text(frame, "Mode: MANUAL (SPACE)  [A]=toggle",
                       (10, 42), font_size=18, color=COLOR_BLUE)

    # State / recording
    if manual_rec.recording:
        pct   = manual_rec.progress / SEQ_LEN
        bar_w = int((w - 20) * pct)
        cv2.rectangle(frame, (10, 95), (10 + bar_w, 104),
                      COLOR_RED, -1)
        cv2.rectangle(frame, (0, 0), (w, h), COLOR_RED, 4)
        put_greek_text(frame,
                       f"REC  {manual_rec.progress}/{SEQ_LEN} frames",
                       (10, 70), font_size=20, color=COLOR_RED)
    elif auto_mode:
        col = COLOR_GREEN if state_auto == "ACTIVE" else COLOR_GRAY
        put_greek_text(frame, f"State: {state_auto}",
                       (10, 70), font_size=18, color=col)

    # ── Τελευταία πρόβλεψη (πάνω αριστερά) ─────────────────
    if last_probs is not None and last_word:
        best_idx  = int(np.argmax(last_probs))
        best_conf = float(last_probs[best_idx])
        gloss     = idx_to_gloss.get(best_idx, str(best_idx))
        put_greek_text_bold(frame,
                            f"{gloss}  {best_conf:.0%}",
                            (10, 110),
                            font_size=28, color=COLOR_GREEN)

    # Motion bar (auto mode)
    if auto_mode:
        bx, by = w - 200, h - 58
        bar_w  = int(min(smooth_motion / 0.1, 1.0) * 110)
        cv2.rectangle(frame, (bx, by), (bx + 110, by + 10),
                      (50, 50, 50), -1)
        col = COLOR_RED if smooth_motion > MOTION_THRESHOLD else COLOR_GRAY
        if bar_w > 0:
            cv2.rectangle(frame, (bx, by),
                          (bx + bar_w, by + 10), col, -1)
        put_greek_text(frame, "motion", (bx, by + 12),
                       font_size=14, color=COLOR_GRAY)

    # ── Κάτω μπάρα ──────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 36), (w, h), (30, 30, 30), -1)
    if last_word:
        put_greek_text(frame,
                       f"Τελευταίο: {last_word} ({last_conf:.0%})",
                       (10, h - 30), font_size=17, color=COLOR_BLUE)
    put_greek_text(frame,
                   f"FPS:{fps:.0f}  C=clear  S=save  Q=quit",
                   (w - 220, h - 30), font_size=16, color=COLOR_GRAY)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--cam",    type=int,   default=DEFAULT_CAM)
    parser.add_argument("--conf",   type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    # ── MediaPipe Tasks API ────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] hand_landmarker.task δεν βρέθηκε: {MODEL_PATH}")
        return

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)
    print("[OK] MediaPipe HandLandmarker φορτώθηκε")

    # ── Μοντέλο ───────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[ERROR] Μοντέλο δεν βρέθηκε: {args.model}")
        return

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt         = torch.load(args.model, map_location=device,
                              weights_only=False)
    idx_to_gloss = ckpt["idx_to_gloss"]
    num_classes  = ckpt["num_classes"]

    model = GSLModel(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[OK] Μοντέλο: {num_classes} κλάσεις | {device}")
    print(f"[OK] Κλάσεις: {list(idx_to_gloss.values())}")

    # ── Webcam ────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Κάμερα {args.cam} δεν ανοίγει")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[OK] Κάμερα ανοιχτή")
    print("     SPACE=εγγραφή | A=auto | C=clear | S=save | Q=quit")

    # ── State ─────────────────────────────────────────────
    segmenter      = GestureSegmenter()
    manual_rec     = ManualRecorder()
    auto_mode      = False
    sentence_words = []
    last_word      = ""
    last_word_time = 0.0
    last_conf      = 0.0
    last_probs     = np.zeros(num_classes)
    fps_deque      = collections.deque(maxlen=30)
    t_prev         = time.time()
    smooth_motion  = 0.0
    state_auto     = "IDLE"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_now = time.time()
        fps_deque.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        fps    = np.mean(fps_deque)

        # MediaPipe στο ORIGINAL frame (ίδιο με extract_keypoints.py)
        kp, result = extract_frame_keypoints(frame, detector)

        # Flip μόνο για display
        display_frame = cv2.flip(frame, 1)
        draw_landmarks_on_frame(display_frame, result)

        # Segmentation
        segment = None
        if manual_rec.recording:
            segment = manual_rec.update(kp)
        elif auto_mode:
            segment, smooth_motion, state_auto = segmenter.update(kp)

        # Inference
        if segment is not None:
            idx, conf, probs = run_inference(model, segment, device)
            last_probs = probs
            if conf >= args.conf:
                gloss = idx_to_gloss[idx]
                now   = time.time()
                if not (gloss == last_word and
                        (now - last_word_time) < DUPLICATE_WINDOW):
                    sentence_words.append(gloss)
                    last_word      = gloss
                    last_word_time = now
                    last_conf      = conf
                    print(f"[GLOSS] {gloss:15s} conf={conf:.2%}"
                          f"  → {' '.join(sentence_words)}")
            else:
                print(f"[LOW]   {idx_to_gloss[idx]:15s} conf={conf:.2%}"
                      f" < {args.conf:.0%}")

        # Display
        draw_ui(display_frame, state_auto, auto_mode, manual_rec,
                smooth_motion, last_word, last_conf,
                sentence_words, last_probs, idx_to_gloss, fps)

        cv2.imshow("GSL Live Recognition", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            if not manual_rec.recording and not auto_mode:
                manual_rec.start()
                print(f"[REC] {SEQ_LEN} frames...")
        elif key == ord('a'):
            auto_mode = not auto_mode
            segmenter._reset()
            print(f"[MODE] {'AUTO' if auto_mode else 'MANUAL'}")
        elif key == ord('c'):
            sentence_words.clear()
            last_word = ""
            print("[CLEAR]")
        elif key == ord('s'):
            fname = f"sentence_{int(time.time())}.txt"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(" ".join(sentence_words))
            print(f"[SAVE] {fname}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\n[DONE]", " ".join(sentence_words))


if __name__ == "__main__":
    main()