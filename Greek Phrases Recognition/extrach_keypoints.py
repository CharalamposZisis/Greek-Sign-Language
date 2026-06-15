"""
GSL Keypoint Extraction από Webcam Videos
==========================================
Εξάγει keypoints από τα .mp4 βίντεο που έγραψε το record_videos.py
χρησιμοποιώντας ΑΚΡΙΒΩΣ το ίδιο pipeline με το extract_keypoints.py
(MediaPipe Tasks API + hand_landmarker.task)

Output: keypoints_webcam/ (.npy files, shape 40x126)
        metadata_webcam.csv

Χρήση:
    python extract_keypoints_videos.py
"""

import os
import cv2
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)


#same paths as extract_keypoints.py

VIDEO_DIRS = [
    "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/webcam_videos",
    "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/Vids_Chris_for_phrases/Vids_Chris_for_phrases",
]
OUT_DIR    = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/keypoints_webcam"
META_OUT   = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/metadata_webcam.csv"
MODEL_PATH = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/hand_landmarker.task"

SEQ_LEN  = 40
NUM_FEAT = 126

GLOSS_TO_IDX = {
    'ΓΕΙΑ':       0,
    'ΕΥΧΑΡΙΣΤΩ':  1,
    'ΠΑΡΑΚΑΛΩ':   2,
    'ΝΑΙ':        3,
    'ΟΧΙ':        4,
    'ΕΝΤΑΞΕΙ':    5,
    'ΘΕΛΩ':       6,
    'ΜΠΟΡΩ':      7,
    'ΧΡΕΙΑΖΟΜΑΙ': 8,
    'ΒΟΗΘΕΙΑ':    9,
}

IDX_TO_GLOSS = {v: k for k, v in GLOSS_TO_IDX.items()}

# MEDIAPIPE SETUP — ίδιο με extract_keypoints.py
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


# KEYPOINT EXTRACTION — ίδιο με extract_keypoints.py
def extract_frame_keypoints(frame_bgr):
    """
    Ακριβώς ίδια συνάρτηση με extract_keypoints.py.
    ΣΗΜΑΝΤΙΚΟ: δεν κάνουμε flip — δίνουμε original frame.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result    = detector.detect(mp_image)

    left_kp  = np.zeros(63, dtype=np.float32)
    right_kp = np.zeros(63, dtype=np.float32)

    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks,
                                         result.handedness):
            label = handedness[0].category_name  # 'Left' or 'Right'
            kp    = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks],
                dtype=np.float32
            ).flatten()
            if label == 'Left':
                right_kp = kp   # ίδια λογική με extract_keypoints.py
            else:
                left_kp  = kp

    return np.concatenate([left_kp, right_kp])  # (126,)


def process_video(video_path):
    """
    Διαβάζει ένα .mp4, εξάγει keypoints από κάθε frame,
    κάνει padding/truncation στα SEQ_LEN frames.
    Επιστρέφει (40, 126) array.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((SEQ_LEN, NUM_FEAT), dtype=np.float32), 0

    keypoints = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # ΣΗΜΑΝΤΙΚΟ: το record_videos.py κάνει flip για display
        # αλλά αποθηκεύει το flipped frame.
        # Οπότε εδώ κάνουμε flip back για να ταιριάζει με GSL.
        frame = cv2.flip(frame, 1)
        kp    = extract_frame_keypoints(frame)
        keypoints.append(kp)

    cap.release()

    if len(keypoints) == 0:
        return np.zeros((SEQ_LEN, NUM_FEAT), dtype=np.float32), 0

    keypoints = np.array(keypoints, dtype=np.float32)
    T         = len(keypoints)

    # Padding / Truncation — ίδιο με extract_keypoints.py
    if T >= SEQ_LEN:
        seq = keypoints[:SEQ_LEN]
    else:
        pad = np.zeros((SEQ_LEN - T, NUM_FEAT), dtype=np.float32)
        seq = np.concatenate([keypoints, pad], axis=0)

    non_zero = int(np.any(seq != 0, axis=1).sum())
    return seq, non_zero



# MAIN

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Δεν βρέθηκε το hand_landmarker.task: {MODEL_PATH}")
        return

    # Βρες όλα τα βίντεο από ΟΛΟΥΣ τους φακέλους
    video_files = []
    for VIDEO_DIR in VIDEO_DIRS:
        if not os.path.exists(VIDEO_DIR):
            print(f"[WARN] Δεν βρέθηκε: {VIDEO_DIR}")
            continue
        source = Path(VIDEO_DIR).name  # π.χ. "webcam_videos" ή "Vids_Chris_for_phrases"
        print(f"[OK] Σάρωση: {VIDEO_DIR}")

        for class_folder in sorted(Path(VIDEO_DIR).iterdir()):
            if not class_folder.is_dir():
                continue
            folder_name = class_folder.name
            if not folder_name.startswith('class'):
                continue
            try:
                class_idx = int(folder_name.split('_')[0].replace('class', ''))
                gloss     = IDX_TO_GLOSS.get(class_idx)
                if gloss is None:
                    continue
            except:
                continue

            for video_file in sorted(class_folder.glob('*.mp4')):
                video_files.append({
                    'path':      video_file,
                    'class_idx': class_idx,
                    'gloss':     gloss,
                    'source':    source,
                })

    if not video_files:
        print(f"[ERROR] Δεν βρέθηκαν .mp4 αρχεία")
        return

    print(f"\n[OK] Βρέθηκαν {len(video_files)} βίντεο συνολικά")
    counts = Counter(v['gloss'] for v in video_files)
    for g, c in sorted(counts.items()):
        print(f"     {g:15s}: {c} βίντεο")

    # Εξαγωγή keypoints
    metadata = []
    errors   = 0
    zero_hand_count = 0

    print(f"\nΕξαγωγή keypoints → {OUT_DIR}/")
    for item in tqdm(video_files, desc="Processing"):
        video_path = item['path']
        class_idx  = item['class_idx']
        gloss      = item['gloss']
        source     = item['source']

        try:
            seq, non_zero = process_video(video_path)

            if non_zero == 0:
                zero_hand_count += 1
                tqdm.write(f"[WARN] Δεν ανιχνεύτηκε χέρι: {video_path.name}")

            safe_name = f"{source}_{video_path.parent.name}_{video_path.stem}.npy"
            npy_path  = os.path.join(OUT_DIR, safe_name)
            np.save(npy_path, seq)

            metadata.append({
                'npy_path':   safe_name,
                'video':      str(video_path.name),
                'gloss':      gloss,
                'label':      class_idx,
                'split':      'train',
                'num_frames': non_zero,
                'source':     source,
            })

        except Exception as e:
            errors += 1
            tqdm.write(f"[ERROR] {video_path.name}: {e}")

    # Αποθήκευση metadata
    fields = ['npy_path', 'video', 'gloss', 'label', 'split', 'num_frames', 'source']
    with open(META_OUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)

    # Summary
    print(f"\n{'='*60}")
    print(f"ΑΠΟΤΕΛΕΣΜΑΤΑ:")
    print(f"  Συνολικά βίντεο:        {len(video_files)}")
    print(f"  Επιτυχής εξαγωγή:       {len(metadata)}")
    print(f"  Errors:                 {errors}")
    print(f"  Χωρίς ανίχνευση χεριού: {zero_hand_count}")

    # Ανά source
    for src in set(m['source'] for m in metadata):
        src_count = sum(1 for m in metadata if m['source'] == src)
        print(f"\n  [{src}]: {src_count} samples")
        for g in GLOSS_TO_IDX:
            c = sum(1 for m in metadata if m['source'] == src and m['gloss'] == g)
            print(f"    {g:15s}: {c}")

    print(f"\n  Output:   {OUT_DIR}")
    print(f"  Metadata: {META_OUT}")

    detector.close()
    
if __name__ == '__main__':
    main()