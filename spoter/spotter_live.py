"""
GSL Live Inference - SPOTER
============================
Πλήκτρα:
    SPACE  : manual trigger (εγγράφει τα επόμενα 40 frames)
    A      : auto mode on/off (motion detection)
    C      : καθαρισμός πρότασης
    S      : αποθήκευση πρότασης
    Q/ESC  : έξοδος
"""

import sys
import argparse
import collections
import time
import os
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)

# ============================================================
# ΡΥΘΜΙΣΕΙΣ — πρώτα ορίζουμε SPOTER_DIR
# ============================================================
SPOTER_DIR      = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/spoter"
CHECKPOINT      = os.path.join(SPOTER_DIR, "out-checkpoints/lsa_64_spoter/checkpoint_v_7.pth")
MEDIAPIPE_MODEL = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/hand_landmarker.task"
DEFAULT_CAM     = 0
CONF_THRESHOLD  = 0.80
SEQ_LEN         = 40
NUM_FEAT        = 126

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

# Normalization imports — μετά τον ορισμό του SPOTER_DIR
sys.path.insert(0, SPOTER_DIR)
from normalization.hand_normalization import normalize_single_dict as normalize_hand
from normalization.body_normalization import normalize_single_dict as normalize_body

IDX_TO_GLOSS = {
    0: 'ΓΕΙΑ',      1: 'ΕΥΧΑΡΙΣΤΩ', 2: 'ΠΑΡΑΚΑΛΩ',
    3: 'ΝΑΙ',       4: 'ΟΧΙ',       5: 'ΕΝΤΑΞΕΙ',
    6: 'ΘΕΛΩ',      7: 'ΜΠΟΡΩ',     8: 'ΧΡΕΙΑΖΟΜΑΙ',
    9: 'ΒΟΗΘΕΙΑ',
}

BODY_IDENTIFIERS = [
    "nose","neck","rightEye","leftEye","rightEar","leftEar",
    "rightShoulder","leftShoulder","rightElbow","leftElbow",
    "rightWrist","leftWrist"
]
HAND_IDENTIFIERS = [
    "wrist","indexTip","indexDIP","indexPIP","indexMCP",
    "middleTip","middleDIP","middlePIP","middleMCP",
    "ringTip","ringDIP","ringPIP","ringMCP",
    "littleTip","littleDIP","littlePIP","littleMCP",
    "thumbTip","thumbIP","thumbMP","thumbCMC"
]
ALL_IDENTIFIERS = (BODY_IDENTIFIERS +
                   [id+"_0" for id in HAND_IDENTIFIERS] +
                   [id+"_1" for id in HAND_IDENTIFIERS])

# MediaPipe → SPOTER landmark mapping
SPOTER_TO_MEDIAPIPE = [
    0,  8,  7,  6,  5,
    12, 11, 10,  9,
    16, 15, 14, 13,
    20, 19, 18, 17,
    4,  3,  2,  1,
]

# ============================================================
# SPOTER MODEL
# ============================================================
def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        del self.self_attn

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=False):
        tgt  = tgt + self.dropout1(tgt)
        tgt  = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt  = tgt + self.dropout2(tgt2)
        tgt  = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt  = tgt + self.dropout3(tgt2)
        tgt  = self.norm3(tgt)
        return tgt


class SPOTER(nn.Module):
    def __init__(self, num_classes, hidden_dim=55):
        super().__init__()
        self.row_embed    = nn.Parameter(torch.rand(50, hidden_dim))
        self.pos          = nn.Parameter(torch.rand(1, 1, hidden_dim))
        self.class_query  = nn.Parameter(torch.rand(1, hidden_dim))
        self.transformer  = nn.Transformer(hidden_dim, 8, 6, 6)
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        custom_decoder_layer = SPOTERTransformerDecoderLayer(
            self.transformer.d_model, self.transformer.nhead, 2048, 0.1, "relu"
        )
        self.transformer.decoder.layers = _get_clones(
            custom_decoder_layer, self.transformer.decoder.num_layers
        )

    def forward(self, inputs):
        h   = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        h   = self.transformer(self.pos + h,
                               self.class_query.unsqueeze(0)).transpose(0, 1)
        return self.linear_class(h)


# ============================================================
# GREEK TEXT
# ============================================================
def put_greek_text(frame, text, pos, font_size=20, color=(255,255,255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def put_greek_text_bold(frame, text, pos, font_size=22, color=(255,255,255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH_BOLD, font_size)
    except:
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except:
            font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# KEYPOINT EXTRACTION
# ============================================================
def extract_frame_keypoints(frame_bgr, detector):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result    = detector.detect(mp_image)
    left_kp   = np.zeros(63, dtype=np.float32)
    right_kp  = np.zeros(63, dtype=np.float32)
    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks,
                                         result.handedness):
            label = handedness[0].category_name
            kp    = np.array([[lm.x, lm.y, lm.z] for lm in landmarks],
                             dtype=np.float32).flatten()
            if label == 'Left':
                right_kp = kp
            else:
                left_kp  = kp
    return np.concatenate([left_kp, right_kp]), result


def npy_to_spoter_tensor(seq):
    """(40, 126) → (40, 54, 2) tensor για SPOTER."""
    T   = seq.shape[0]
    out = np.zeros((T, 54, 2), dtype=np.float32)
    for spoter_idx, mp_idx in enumerate(SPOTER_TO_MEDIAPIPE):
        # Left hand → indices 12-32
        out[:, 12 + spoter_idx, 0] = seq[:, mp_idx * 3]
        out[:, 12 + spoter_idx, 1] = seq[:, mp_idx * 3 + 1]
        # Right hand → indices 33-53
        out[:, 33 + spoter_idx, 0] = seq[:, 63 + mp_idx * 3]
        out[:, 33 + spoter_idx, 1] = seq[:, 63 + mp_idx * 3 + 1]
    return torch.FloatTensor(out)  # (40, 54, 2)


def tensor_to_dict(tensor):
    """(40, 54, 2) tensor → dict για normalization."""
    data = tensor.numpy()
    row  = {}
    for i, ident in enumerate(ALL_IDENTIFIERS):
        row[ident] = [data[t, i].tolist() for t in range(data.shape[0])]
    return row


def dict_to_tensor(row):
    """dict → (40, 54, 2) numpy."""
    out = np.zeros((40, 54, 2), dtype=np.float32)
    for i, ident in enumerate(ALL_IDENTIFIERS):
        for t in range(40):
            val = row[ident][t]
            if isinstance(val, (list, np.ndarray)):
                out[t, i, 0] = val[0]
                out[t, i, 1] = val[1]
            else:
                out[t, i, 0] = float(val)
    return out


# ============================================================
# INFERENCE
# ============================================================
@torch.no_grad()
def run_inference(model, sequence, device):
    """sequence: numpy (40, 126)"""
    # Μετατροπή σε SPOTER format
    spoter_seq = npy_to_spoter_tensor(sequence)  # (40, 54, 2)

    # Normalization — ίδιο με CzechSLRDataset
    row = tensor_to_dict(spoter_seq)
    row = normalize_body(row)
    row = normalize_hand(row)

    # Πίσω σε tensor
    out = dict_to_tensor(row)
    x   = torch.FloatTensor(out).to(device)     # (40, 54, 2)

    logits = model(x)                            # (1, 1, 10)
    logits = logits.squeeze(0).squeeze(0)        # (10,)
    probs  = torch.softmax(logits, dim=0).cpu().numpy()
    idx    = int(np.argmax(probs))
    return idx, float(probs[idx]), probs


# ============================================================
# MOTION DETECTION
# ============================================================
def compute_motion(prev_kp, curr_kp):
    mask = (prev_kp != 0) & (curr_kp != 0)
    if mask.sum() == 0:
        return 0.0
    return float(np.abs(curr_kp[mask] - prev_kp[mask]).mean())


class GestureSegmenter:
    IDLE = "IDLE"; ACTIVE = "ACTIVE"

    def __init__(self):
        self.state          = self.IDLE
        self.buffer         = []
        self.still_count    = 0
        self.motion_history = collections.deque(maxlen=MOTION_WINDOW)
        self.prev_kp        = None

    def update(self, kp):
        segment = None
        motion  = compute_motion(self.prev_kp, kp) if self.prev_kp is not None else 0.0
        self.motion_history.append(motion)
        self.prev_kp  = kp.copy()
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
            self.still_count = 0 if (is_moving and has_hands) else self.still_count + 1
            if self.still_count >= STILLNESS_FRAMES or len(self.buffer) >= SEQ_LEN:
                if len(self.buffer) >= MIN_ACTIVE_FRAMES:
                    arr = np.array(self.buffer, dtype=np.float32)
                    T   = len(arr)
                    if T >= SEQ_LEN:
                        segment = arr[:SEQ_LEN]
                    else:
                        pad     = np.zeros((SEQ_LEN-T, NUM_FEAT), dtype=np.float32)
                        segment = np.vstack([arr, pad])
                self._reset()

        return segment, smooth_motion, self.state

    def _reset(self):
        self.state       = self.IDLE
        self.buffer      = []
        self.still_count = 0


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
            seq            = np.array(self.buffer, dtype=np.float32)
            self.recording = False
            self.buffer    = []
            return seq
        return None

    @property
    def progress(self):
        return len(self.buffer)


# ============================================================
# DISPLAY
# ============================================================
def draw_landmarks_on_frame(frame, result):
    if not result.hand_landmarks:
        return
    h, w        = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
    ]
    for hand_landmarks in result.hand_landmarks:
        pts = [(int((1-lm.x)*w), int(lm.y*h)) for lm in hand_landmarks]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0,100,255), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0,220,0), -1)


def draw_ui(frame, state_auto, auto_mode, manual_rec, smooth_motion,
            last_word, last_conf, sentence_words, last_probs, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,105), (30,30,30), -1)
    text = " ".join(sentence_words) if sentence_words else "..."
    put_greek_text_bold(frame, text, (10,5), font_size=26, color=COLOR_GREEN)

    if auto_mode:
        put_greek_text(frame, "Mode: AUTO  [A]=toggle",
                       (10,42), font_size=18, color=COLOR_YELLOW)
    else:
        put_greek_text(frame, "Mode: MANUAL (SPACE)  [A]=toggle",
                       (10,42), font_size=18, color=COLOR_BLUE)

    if manual_rec.recording:
        pct   = manual_rec.progress / SEQ_LEN
        bar_w = int((w-20)*pct)
        cv2.rectangle(frame, (10,95), (10+bar_w,104), COLOR_RED, -1)
        cv2.rectangle(frame, (0,0), (w,h), COLOR_RED, 4)
        put_greek_text(frame, f"REC  {manual_rec.progress}/{SEQ_LEN} frames",
                       (10,70), font_size=20, color=COLOR_RED)
    elif auto_mode:
        col = COLOR_GREEN if state_auto == "ACTIVE" else COLOR_GRAY
        put_greek_text(frame, f"State: {state_auto}",
                       (10,70), font_size=18, color=col)

    if last_probs is not None and last_word:
        best_idx  = int(np.argmax(last_probs))
        best_conf = float(last_probs[best_idx])
        gloss     = IDX_TO_GLOSS.get(best_idx, str(best_idx))
        put_greek_text_bold(frame, f"{gloss}  {best_conf:.0%}",
                            (10,110), font_size=28, color=COLOR_GREEN)

    if auto_mode:
        bx, by = w-200, h-58
        bar_w  = int(min(smooth_motion/0.1,1.0)*110)
        cv2.rectangle(frame, (bx,by), (bx+110,by+10), (50,50,50), -1)
        col = COLOR_RED if smooth_motion > MOTION_THRESHOLD else COLOR_GRAY
        if bar_w > 0:
            cv2.rectangle(frame, (bx,by), (bx+bar_w,by+10), col, -1)
        put_greek_text(frame, "motion", (bx,by+12), font_size=14, color=COLOR_GRAY)

    cv2.rectangle(frame, (0,h-36), (w,h), (30,30,30), -1)
    if last_word:
        put_greek_text(frame, f"Τελευταίο: {last_word} ({last_conf:.0%})",
                       (10,h-30), font_size=17, color=COLOR_BLUE)
    put_greek_text(frame, f"FPS:{fps:.0f}  C=clear  S=save  Q=quit",
                   (w-220,h-30), font_size=16, color=COLOR_GRAY)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--cam",  type=int,   default=DEFAULT_CAM)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    if not os.path.exists(MEDIAPIPE_MODEL):
        print(f"[ERROR] hand_landmarker.task δεν βρέθηκε: {MEDIAPIPE_MODEL}")
        return

    base_options = mp_python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL)
    options = HandLandmarkerOptions(
        base_options=base_options, running_mode=RunningMode.IMAGE,
        num_hands=2, min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5, min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)
    print("[OK] MediaPipe HandLandmarker φορτώθηκε")

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint δεν βρέθηκε: {args.checkpoint}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.eval()
    print(f"[OK] SPOTER φορτώθηκε | {device}")
    print(f"[OK] Κλάσεις: {list(IDX_TO_GLOSS.values())}")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Κάμερα {args.cam} δεν ανοίγει")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[OK] Κάμερα ανοιχτή")
    print("     SPACE=εγγραφή | A=auto | C=clear | S=save | Q=quit")

    segmenter      = GestureSegmenter()
    manual_rec     = ManualRecorder()
    auto_mode      = False
    sentence_words = []
    last_word      = ""
    last_word_time = 0.0
    last_conf      = 0.0
    last_probs     = np.zeros(10)
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

        kp, result    = extract_frame_keypoints(frame, detector)
        display_frame = cv2.flip(frame, 1)
        draw_landmarks_on_frame(display_frame, result)

        segment = None
        if manual_rec.recording:
            segment = manual_rec.update(kp)
        elif auto_mode:
            segment, smooth_motion, state_auto = segmenter.update(kp)

        if segment is not None:
            idx, conf, probs = run_inference(model, segment, device)
            last_probs = probs
            # SPOTER labels 1-indexed → IDX_TO_GLOSS 0-indexed
            gloss_idx = idx
            gloss     = IDX_TO_GLOSS.get(gloss_idx, str(gloss_idx))
            if conf >= args.conf:
                now = time.time()
                if not (gloss == last_word and
                        (now - last_word_time) < DUPLICATE_WINDOW):
                    sentence_words.append(gloss)
                    last_word      = gloss
                    last_word_time = now
                    last_conf      = conf
                    print(f"[GLOSS] {gloss:15s} conf={conf:.2%}"
                          f"  → {' '.join(sentence_words)}")
            else:
                print(f"[LOW]   {gloss:15s} conf={conf:.2%}")

        draw_ui(display_frame, state_auto, auto_mode, manual_rec,
                smooth_motion, last_word, last_conf,
                sentence_words, last_probs, fps)

        cv2.imshow("GSL SPOTER Live", display_frame)

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