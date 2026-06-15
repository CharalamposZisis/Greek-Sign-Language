"""
GSL Webcam Video Recorder
=========================
Εγγράφει βίντεο 40 frames ανά νόημα από webcam.
Μετά τρέξε extract_keypoints_videos.py για να εξάγεις τα keypoints.

Πλήκτρα:
    SPACE  : ξεκινά εγγραφή (μετράει αντίστροφα 3-2-1)
    N      : επόμενη κλάση
    P      : προηγούμενη κλάση
    D      : διαγραφή τελευταίου βίντεο
    Q/ESC  : έξοδος
"""

import os
import cv2
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# ============================================================
# ΡΥΘΜΙΣΕΙΣ
# ============================================================
OUTPUT_DIR        = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/webcam_videos"
SAMPLES_PER_CLASS = 40
FRAMES_PER_VIDEO  = 40
FPS               = 20
COUNTDOWN_SEC     = 3
CAM_INDEX         = 0
FRAME_W           = 640
FRAME_H           = 480

GLOSSES = {
    0: 'ΓΕΙΑ',
    1: 'ΕΥΧΑΡΙΣΤΩ',
    2: 'ΠΑΡΑΚΑΛΩ',
    3: 'ΝΑΙ',
    4: 'ΟΧΙ',
    5: 'ΕΝΤΑΞΕΙ',
    6: 'ΘΕΛΩ',
    7: 'ΜΠΟΡΩ',
    8: 'ΧΡΕΙΑΖΟΜΑΙ',
    9: 'ΒΟΗΘΕΙΑ',
}

COLOR_GREEN  = (0, 220, 0)
COLOR_RED    = (0, 0, 220)
COLOR_BLUE   = (255, 160, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (160, 160, 160)
COLOR_YELLOW = (0, 220, 220)

# Font για ελληνικά
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ============================================================
# GREEK TEXT με PIL
# ============================================================
def put_greek_text(frame, text, pos, font_size=22, color=(255, 255, 255)):
    """Γράφει ελληνικό κείμενο με PIL — color σε BGR."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    # Μετατροπή BGR → RGB για PIL
    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def put_greek_text_bold(frame, text, pos, font_size=24, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH_BOLD, font_size)
    except:
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except:
            font = ImageFont.load_default()
    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# HELPERS
# ============================================================
def class_dir(class_idx):
    gloss = GLOSSES[class_idx]
    return os.path.join(OUTPUT_DIR, f"class{class_idx:02d}_{gloss}")


def get_video_count(class_idx):
    d = class_dir(class_idx)
    if not os.path.exists(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith('.mp4')])


def next_video_path(class_idx):
    d   = class_dir(class_idx)
    os.makedirs(d, exist_ok=True)
    idx = get_video_count(class_idx)
    return os.path.join(d, f"sample_{idx:03d}.mp4")


def delete_last_video(class_idx):
    d = class_dir(class_idx)
    if not os.path.exists(d):
        return False
    files = sorted([f for f in os.listdir(d) if f.endswith('.mp4')])
    if files:
        path = os.path.join(d, files[-1])
        os.remove(path)
        print(f"[DELETE] {files[-1]}")
        return True
    return False


def draw_progress_bar(frame, current, total, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    filled = int(w * current / max(total, 1))
    color  = COLOR_GREEN if current >= total else COLOR_BLUE
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_ui(frame, class_idx, state, countdown_val, frames_recorded):
    h, w  = frame.shape[:2]
    gloss = GLOSSES[class_idx]
    count = get_video_count(class_idx)

    # ── Πάνω μπάρα ──────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 105), (30, 30, 30), -1)

    # Κλάση + gloss
    put_greek_text_bold(frame,
                        f"Κλάση {class_idx}:  {gloss}",
                        (15, 8), font_size=26,
                        color=COLOR_YELLOW)

    # Progress bar
    draw_progress_bar(frame, count, SAMPLES_PER_CLASS,
                      x=15, y=50, w=w - 80, h=18)
    put_greek_text(frame, f"{count}/{SAMPLES_PER_CLASS}",
                   (w - 60, 50), font_size=18, color=COLOR_WHITE)

    # State message
    if state == 'IDLE':
        if count >= SAMPLES_PER_CLASS:
            msg = "✓ ΟΛΟΚΛΗΡΩΘΗΚΕ!   N = επομενη κλαση"
            col = COLOR_GREEN
        else:
            msg = "SPACE=εγγραφη  N/P=αλλαγη  D=διαγραφη"
            col = COLOR_GRAY
        put_greek_text(frame, msg, (15, 78), font_size=18, color=col)

    elif state == 'COUNTDOWN':
        put_greek_text(frame,
                       f"Ετοιμάσου...  {countdown_val}",
                       (15, 78), font_size=20, color=COLOR_YELLOW)

    elif state == 'RECORDING':
        put_greek_text(frame,
                       f"REC  {frames_recorded}/{FRAMES_PER_VIDEO} frames",
                       (15, 78), font_size=20, color=COLOR_RED)
        cv2.rectangle(frame, (0, 0), (w, h), COLOR_RED, 5)
        # Μπάρα προόδου εγγραφής (δεξιά)
        pct      = frames_recorded / FRAMES_PER_VIDEO
        bar_fill = int((h - 110 - 38) * pct)
        cv2.rectangle(frame, (w - 18, 110),
                      (w - 5, h - 38), (60, 60, 60), -1)
        if bar_fill > 0:
            cv2.rectangle(frame, (w - 18, h - 38 - bar_fill),
                          (w - 5, h - 38), COLOR_RED, -1)

    # ── Κάτω μπάρα ──────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 36), (w, h), (30, 30, 30), -1)
    total_done = sum(get_video_count(i) for i in GLOSSES)
    total_need = len(GLOSSES) * SAMPLES_PER_CLASS
    put_greek_text(frame,
                   f"Συνολο: {total_done}/{total_need}",
                   (15, h - 30), font_size=17, color=COLOR_GRAY)
    put_greek_text(frame, "Q/ESC=εξοδος",
                   (w - 130, h - 30), font_size=17, color=COLOR_GRAY)

    # Countdown overlay (μεγάλο νούμερο)
    if state == 'COUNTDOWN' and countdown_val > 0:
        cv2.putText(frame, str(countdown_val),
                    (w // 2 - 40, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 5.0, COLOR_YELLOW, 10)


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Δεν ανοίγει η κάμερα {CAM_INDEX}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print("=" * 60)
    print("GSL Webcam Video Recorder")
    print("=" * 60)
    for i, g in GLOSSES.items():
        done = get_video_count(i)
        print(f"  Κλάση {i}: {g:15s} → {done}/{SAMPLES_PER_CLASS}")

    # Βρες πρώτη ημιτελή κλάση
    class_idx = 0
    for i in GLOSSES:
        if get_video_count(i) < SAMPLES_PER_CLASS:
            class_idx = i
            break

    state           = 'IDLE'
    countdown_start = 0.0
    frames_recorded = 0
    writer          = None
    video_path      = None

    print(f"\n[START] Κλάση {class_idx}: {GLOSSES[class_idx]}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip για φυσική εμφάνιση
        frame = cv2.flip(frame, 1)

        # ── State logic ──────────────────────────────────────
        if state == 'COUNTDOWN':
            elapsed   = time.time() - countdown_start
            remaining = COUNTDOWN_SEC - int(elapsed)
            if elapsed >= COUNTDOWN_SEC:
                state           = 'RECORDING'
                frames_recorded = 0
                video_path      = next_video_path(class_idx)
                writer          = cv2.VideoWriter(
                    video_path, fourcc, FPS, (FRAME_W, FRAME_H))
                print(f"[REC] {os.path.basename(video_path)}")
            draw_ui(frame, class_idx, 'COUNTDOWN',
                    max(remaining, 1), 0)

        elif state == 'RECORDING':
            writer.write(frame)
            frames_recorded += 1
            draw_ui(frame, class_idx, 'RECORDING', 0, frames_recorded)

            if frames_recorded >= FRAMES_PER_VIDEO:
                writer.release()
                writer = None
                count  = get_video_count(class_idx)
                print(f"[SAVE] {os.path.basename(video_path)}"
                      f"  ({count}/{SAMPLES_PER_CLASS})")
                state = 'IDLE'
                if count >= SAMPLES_PER_CLASS:
                    print(f"[DONE] {GLOSSES[class_idx]} ολοκληρώθηκε!")

        else:  # IDLE
            draw_ui(frame, class_idx, 'IDLE', 0, 0)

        cv2.imshow("GSL Video Recorder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' ') and state == 'IDLE':
            if get_video_count(class_idx) >= SAMPLES_PER_CLASS:
                print(f"[INFO] {GLOSSES[class_idx]} ηδη ολοκληρωμενη!")
            else:
                state           = 'COUNTDOWN'
                countdown_start = time.time()
                print(f"[COUNTDOWN] {COUNTDOWN_SEC}...")
        elif key == ord('n') and state == 'IDLE':
            class_idx = (class_idx + 1) % len(GLOSSES)
            print(f"[NEXT] {class_idx}: {GLOSSES[class_idx]}"
                  f" ({get_video_count(class_idx)}/{SAMPLES_PER_CLASS})")
        elif key == ord('p') and state == 'IDLE':
            class_idx = (class_idx - 1) % len(GLOSSES)
            print(f"[PREV] {class_idx}: {GLOSSES[class_idx]}"
                  f" ({get_video_count(class_idx)}/{SAMPLES_PER_CLASS})")
        elif key == ord('d') and state == 'IDLE':
            delete_last_video(class_idx)

    if writer is not None:
        writer.release()
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            print("[INFO] Ημιτελες βιντεο διαγραφηκε")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("ΣΥΝΟΨΗ:")
    total = 0
    for i, g in GLOSSES.items():
        done   = get_video_count(i)
        total += done
        status = "OK" if done >= SAMPLES_PER_CLASS else f"{done}/{SAMPLES_PER_CLASS}"
        print(f"  {g:15s}: {status}")
    print(f"\nΣυνολο: {total} βιντεο")
    print(f"Output: {OUTPUT_DIR}")
    print("\nΕπομενο βημα: extract_keypoints_videos.py")


if __name__ == "__main__":
    main()