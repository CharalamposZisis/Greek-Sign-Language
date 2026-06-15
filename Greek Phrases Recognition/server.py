
"""
GSL SPOTER Server — HTTP Polling (no WebSocket issues)
=======================================================
- Μετάφραση μέσω HTTP polling (δουλεύει παντού, κινητά, Cloudflare)
- Κάμερα viewer μέσω HTTP POST
- MJPEG stream εσένα → browser

Τρέξε:
    python server.py
Μετά:
    cloudflared tunnel --url http://localhost:8000
Για να ανοιξει το tunnel εκτελουμε στο cmd --> # cloudflared tunnel --url http://localhost:8000  #
"""

import sys, os, copy, threading, time, base64
import numpy as np
import cv2
import torch
import torch.nn as nn

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

# ============================================================
# ΡΥΘΜΙΣΕΙΣ
# ============================================================
SPOTER_DIR      = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/spoter"
CHECKPOINT      = os.path.join(SPOTER_DIR, "out-checkpoints/lsa_64_spoter/checkpoint_v_7.pth")
MEDIAPIPE_MODEL = "/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/21/05/2026/hand_landmarker.task"
CAM_INDEX       = 0
CONF_THRESHOLD  = 0.60
SEQ_LEN         = 40
HOST            = "0.0.0.0"
PORT            = 8000

MJPEG_FPS     = 8
MJPEG_QUALITY = 35

sys.path.insert(0, SPOTER_DIR)
from normalization.hand_normalization import normalize_single_dict as normalize_hand
from normalization.body_normalization import normalize_single_dict as normalize_body

IDX_TO_GLOSS = {
    0:'ΓΕΙΑ', 1:'ΕΥΧΑΡΙΣΤΩ', 2:'ΠΑΡΑΚΑΛΩ', 3:'ΝΑΙ', 4:'ΟΧΙ',
    5:'ΕΝΤΑΞΕΙ', 6:'ΘΕΛΩ', 7:'ΜΠΟΡΩ', 8:'ΧΡΕΙΑΖΟΜΑΙ', 9:'ΒΟΗΘΕΙΑ', 
}

BODY_IDENTIFIERS = ["nose","neck","rightEye","leftEye","rightEar","leftEar",
    "rightShoulder","leftShoulder","rightElbow","leftElbow","rightWrist","leftWrist"]
HAND_IDENTIFIERS = ["wrist","indexTip","indexDIP","indexPIP","indexMCP",
    "middleTip","middleDIP","middlePIP","middleMCP","ringTip","ringDIP","ringPIP",
    "ringMCP","littleTip","littleDIP","littlePIP","littleMCP",
    "thumbTip","thumbIP","thumbMP","thumbCMC"]
ALL_IDENTIFIERS = (BODY_IDENTIFIERS +
    [id+"_0" for id in HAND_IDENTIFIERS] + [id+"_1" for id in HAND_IDENTIFIERS])
SPOTER_TO_MEDIAPIPE = [0,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,4,3,2,1]

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
        custom = SPOTERTransformerDecoderLayer(
            self.transformer.d_model, self.transformer.nhead, 2048, 0.1, "relu")
        self.transformer.decoder.layers = _get_clones(
            custom, self.transformer.decoder.num_layers)

    def forward(self, inputs):
        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        h = self.transformer(self.pos + h,
                             self.class_query.unsqueeze(0)).transpose(0, 1)
        return self.linear_class(h)

# ============================================================
# KEYPOINTS + INFERENCE
# ============================================================
def extract_frame_keypoints(frame_bgr, detector):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result    = detector.detect(mp_image)
    left_kp   = np.zeros(63, dtype=np.float32)
    right_kp  = np.zeros(63, dtype=np.float32)
    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            kp    = np.array([[lm.x,lm.y,lm.z] for lm in landmarks],
                             dtype=np.float32).flatten()
            if label == 'Left': right_kp = kp
            else:               left_kp  = kp
    return np.concatenate([left_kp, right_kp]), result

def npy_to_spoter_tensor(seq):
    T   = seq.shape[0]
    out = np.zeros((T, 54, 2), dtype=np.float32)
    for si, mi in enumerate(SPOTER_TO_MEDIAPIPE):
        out[:,12+si,0]=seq[:,mi*3];    out[:,12+si,1]=seq[:,mi*3+1]
        out[:,33+si,0]=seq[:,63+mi*3]; out[:,33+si,1]=seq[:,63+mi*3+1]
    return torch.FloatTensor(out)

def tensor_to_dict(tensor):
    data = tensor.numpy()
    return {ident:[data[t,i].tolist() for t in range(data.shape[0])]
            for i,ident in enumerate(ALL_IDENTIFIERS)}

def dict_to_array(row):
    out = np.zeros((40,54,2), dtype=np.float32)
    for i,ident in enumerate(ALL_IDENTIFIERS):
        for t in range(40):
            val = row[ident][t]
            if isinstance(val,(list,np.ndarray)):
                out[t,i,0]=val[0]; out[t,i,1]=val[1]
            else: out[t,i,0]=float(val)
    return out

@torch.no_grad()
def run_inference(model, sequence, device):
    spoter_seq = npy_to_spoter_tensor(sequence)
    row = tensor_to_dict(spoter_seq)
    row = normalize_body(row); row = normalize_hand(row)
    out = dict_to_array(row)
    x   = torch.FloatTensor(out).to(device)
    logits = model(x).squeeze(0).squeeze(0)
    probs  = torch.softmax(logits, dim=0).cpu().numpy()
    idx    = int(np.argmax(probs))
    return idx, float(probs[idx])

def draw_landmarks(frame, result):
    if not result.hand_landmarks: return
    h,w = frame.shape[:2]
    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
             (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    for hl in result.hand_landmarks:
        pts = [(int((1-lm.x)*w),int(lm.y*h)) for lm in hl]
        for a,b in conns: cv2.line(frame,pts[a],pts[b],(0,100,255),2)
        for pt in pts:    cv2.circle(frame,pt,4,(0,220,0),-1)

# ============================================================
# GLOBAL STATE — thread-safe με locks
# ============================================================
class State:
    def __init__(self):
        self.lock         = threading.Lock()
        self.sentence     = []
        self.last_gloss   = ""
        self.last_conf    = 0.0
        self.update_id    = 0      # αυξάνεται με κάθε νέα πρόβλεψη
        self.frame_lock   = threading.Lock()
        self.latest_frame = None   # MJPEG εσένα
        self.viewer_lock  = threading.Lock()
        self.viewer_frame = None   # base64 frame άλλου

state = State()

# ============================================================
# HTML
# ============================================================
HTML = """<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GSL Live</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;color:#fff;
     font-family:'Segoe UI',sans-serif;
     min-height:100vh;display:flex;flex-direction:column;
     align-items:center;padding:16px;gap:14px}
h1{font-size:1rem;color:#444;letter-spacing:3px;text-transform:uppercase}
#status{font-size:.8rem;padding:4px 14px;border-radius:20px;
        background:#1a1a1a;color:#555}
#status.on{color:#00dc82;background:#001a0e}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;
      width:100%;max-width:860px}
.vbox{display:flex;flex-direction:column;gap:6px}
.vlabel{font-size:.7rem;color:#444;letter-spacing:2px;
        text-transform:uppercase;text-align:center}
.vwrap{border-radius:12px;overflow:hidden;border:1px solid #1e1e1e;
       background:#111;aspect-ratio:4/3;display:flex;
       align-items:center;justify-content:center;position:relative}
.vwrap img,.vwrap video{width:100%;height:100%;object-fit:cover;display:block}
#my-video{transform:scaleX(-1)}
#viewer-img{transform:scaleX(-1)}
#cam-status{font-size:.75rem;color:#555;text-align:center}
#cam-status.on{color:#00dc82}
.placeholder{color:#333;font-size:.8rem;text-align:center;padding:8px}
.trans{width:100%;max-width:860px;display:flex;flex-direction:column;gap:10px}
#last-word{font-size:clamp(2.5rem,8vw,5rem);font-weight:700;
           color:#00dc82;text-align:center;min-height:1.2em;
           text-shadow:0 0 30px rgba(0,220,130,.3);transition:all .3s}
#conf{font-size:1rem;color:#333;text-align:center;letter-spacing:3px}
#sentence-box{background:#111;border:1px solid #1e1e1e;border-radius:10px;
              padding:12px;min-height:60px;display:flex;flex-wrap:wrap;
              gap:7px;align-items:flex-start}
.chip{background:#1a1a2e;border:1px solid #2a2a3e;border-radius:6px;
      padding:4px 10px;font-size:.9rem;color:#888}
.chip.new{background:#002a18;border-color:#00dc82;color:#00dc82}
.btns{display:flex;gap:10px}
.btn{flex:1;padding:9px;border-radius:8px;cursor:pointer;
     font-size:.85rem;border:1px solid #2a2a2a;
     background:#1a1a1a;color:#555;transition:all .2s;
     -webkit-tap-highlight-color:transparent}
.btn.green{border-color:#00dc82;background:#001a0e;color:#00dc82}
</style>
</head>
<body>
<h1>Greek Sign Language — Live</h1>
<div id="status">Connecting...</div>

<div class="grid">
  <div class="vbox">
    <div class="vlabel">Signer</div>
    <div class="vwrap">
      <img id="signer-img" src="/stream" alt="Signer"
           onerror="this.style.display='none'">
    </div>
  </div>
  <div class="vbox">
    <div class="vlabel">You</div>
    <div class="vwrap" id="you-wrap">
      <video id="my-video" autoplay playsinline muted
             style="display:none"></video>
      <img id="viewer-img" style="display:none" alt="Viewer">
      <div class="placeholder" id="ph">Enable camera below</div>
    </div>
    <div id="cam-status">Camera off</div>
    <button class="btn green" id="cam-btn" onclick="toggleCamera()">
      Enable Camera
    </button>
  </div>
</div>

<div class="trans">
  <div id="last-word">...</div>
  <div id="conf"></div>
  <div id="sentence-box"></div>
  <div class="btns">
    <button class="btn" onclick="clearAll()">Clear</button>
  </div>
</div>

<script>
let lastId    = 0;
let camOn     = false;
let camStream = null;
let sendIv    = null;
let pollIv    = null;
let viewerIv  = null;

// ── Status ──────────────────────────────────────────────────
document.getElementById('status').textContent = 'Connected';
document.getElementById('status').className   = 'on';

// ── Poll για μετάφραση κάθε 1.5 sec ────────────────────────
async function pollTranslation(){
  try{
    const r = await fetch(`/translation?since=${lastId}`);
    if(!r.ok) return;
    const d = await r.json();
    if(d.id > lastId){
      lastId = d.id;
      // Τελευταία λέξη
      document.getElementById('last-word').textContent = d.gloss;
      document.getElementById('conf').textContent =
        Math.round(d.confidence*100)+'%';
      const g = Math.round(d.confidence*220);
      document.getElementById('last-word').style.color =
        `rgb(0,${g},${Math.round(g*.6)})`;
      // Πρόταση
      const box = document.getElementById('sentence-box');
      box.innerHTML='';
      d.sentence.forEach((w,i)=>{
        const c=document.createElement('div');
        c.className='chip'+(i===d.sentence.length-1?' new':'');
        c.textContent=w; box.appendChild(c);
      });
    }
  }catch(e){}
}
pollIv = setInterval(pollTranslation, 1500);
pollTranslation();  // αμέσως

// ── Poll για κάμερα viewer (μόνο αν ΔΕΝ έχεις κάμερα) ──────
async function pollViewer(){
  if(camOn) return;  // αν έχεις κάμερα, εσύ είσαι viewer
  try{
    const r = await fetch('/viewer-frame');
    if(!r.ok) return;
    const d = await r.json();
    if(d.data){
      document.getElementById('ph').style.display='none';
      document.getElementById('my-video').style.display='none';
      const vimg=document.getElementById('viewer-img');
      vimg.style.display='block';
      vimg.src=d.data;
    }
  }catch(e){}
}
viewerIv = setInterval(pollViewer, 500);

// ── Κάμερα ──────────────────────────────────────────────────
async function toggleCamera(){
  if(!camOn){
    try{
      camStream = await navigator.mediaDevices.getUserMedia(
                    {video:{facingMode:'user'}, audio:false});
      const vid = document.getElementById('my-video');
      vid.srcObject = camStream;
      document.getElementById('ph').style.display='none';
      document.getElementById('viewer-img').style.display='none';
      vid.style.display='block';
      document.getElementById('cam-status').textContent='Camera on';
      document.getElementById('cam-status').className='on';
      document.getElementById('cam-btn').textContent='Disable Camera';
      camOn=true;

      const canvas=document.createElement('canvas');
      canvas.width=160; canvas.height=120;
      const ctx=canvas.getContext('2d');

      // Στέλνε frame κάθε 800ms (1.25fps — χαμηλό bandwidth)
      sendIv=setInterval(async ()=>{
        try{
          ctx.drawImage(vid,0,0,160,120);
          const data=canvas.toDataURL('image/jpeg',.35);
          await fetch('/viewer-frame',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({data})
          });
        }catch(e){}
      }, 800);

    }catch(e){
      document.getElementById('cam-status').textContent='Error: '+e.message;
      alert('Camera error: '+e.message);
    }
  } else {
    if(camStream) camStream.getTracks().forEach(t=>t.stop());
    if(sendIv) clearInterval(sendIv);
    const vid=document.getElementById('my-video');
    vid.srcObject=null; vid.style.display='none';
    document.getElementById('viewer-img').style.display='none';
    document.getElementById('ph').style.display='flex';
    document.getElementById('cam-status').textContent='Camera off';
    document.getElementById('cam-status').className='';
    document.getElementById('cam-btn').textContent='Enable Camera';
    camOn=false;
  }
}

async function clearAll(){
  try{ await fetch('/clear',{method:'POST'}); }catch(e){}
  document.getElementById('sentence-box').innerHTML='';
  document.getElementById('last-word').textContent='...';
  document.getElementById('conf').textContent='';
  lastId=0;
}
</script>
</body>
</html>"""

# ============================================================
# FASTAPI
# ============================================================
app = FastAPI()

@app.get("/")
async def index():
    return HTMLResponse(HTML)

# ── MJPEG stream ──────────────────────────────────────────
def mjpeg_generator():
    interval = 1.0 / MJPEG_FPS
    while True:
        with state.frame_lock:
            frame = state.latest_frame
        if frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + frame + b"\r\n")
        time.sleep(interval)

@app.get("/stream")
async def video_stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame")

# ── Translation polling ───────────────────────────────────
@app.get("/translation")
async def get_translation(since: int = 0):
    with state.lock:
        return JSONResponse({
            "id":         state.update_id,
            "gloss":      state.last_gloss,
            "confidence": state.last_conf,
            "sentence":   state.sentence.copy(),
        })

# ── Clear ─────────────────────────────────────────────────
@app.post("/clear")
async def clear():
    with state.lock:
        state.sentence   = []
        state.last_gloss = ""
        state.last_conf  = 0.0
        state.update_id  += 1
    return JSONResponse({"ok": True})

# ── Viewer frame POST (viewer → server) ───────────────────
@app.post("/viewer-frame")
async def post_viewer_frame(request: Request):
    body = await request.json()
    data = body.get("data","")
    with state.viewer_lock:
        state.viewer_frame = data
    return JSONResponse({"ok": True})

# ── Viewer frame GET (server → signer) ───────────────────
@app.get("/viewer-frame")
async def get_viewer_frame():
    with state.viewer_lock:
        data = state.viewer_frame
    return JSONResponse({"data": data})

# ============================================================
# INFERENCE LOOP
# ============================================================
def inference_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.eval()
    print(f"[OK] SPOTER | {device}")

    base_options = mp_python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL)
    options = HandLandmarkerOptions(
        base_options=base_options, running_mode=RunningMode.IMAGE,
        num_hands=2, min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5, min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)
    print("[OK] MediaPipe")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[OK] Καμερα")
    print("     SPACE=εγγραφη | Q=εξοδος")

    buffer    = []
    recording = False

    while True:
        ret, frame = cap.read()
        if not ret: break

        display = cv2.flip(frame, 1)
        kp, result = extract_frame_keypoints(frame, detector)
        draw_landmarks(display, result)

        if recording:
            buffer.append(kp.copy())
            if len(buffer) >= SEQ_LEN:
                recording = False
                seq       = np.array(buffer, dtype=np.float32)
                buffer    = []

                idx, conf = run_inference(model, seq, device)
                gloss     = IDX_TO_GLOSS.get(idx, str(idx))
                print(f"[GLOSS] {gloss} ({conf:.0%})")

                with state.lock:
                    if conf >= CONF_THRESHOLD:
                        state.sentence.append(gloss)
                    state.last_gloss = gloss
                    state.last_conf  = conf
                    state.update_id += 1

        # Encode → MJPEG
        _, jpeg = cv2.imencode('.jpg', display,
                               [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        with state.frame_lock:
            state.latest_frame = jpeg.tobytes()

        lbl = "REC" if recording else "READY"
        col = (0,0,220) if recording else (0,200,0)
        cv2.putText(display, lbl, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.imshow("GSL Server", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        elif key == ord(' ') and not recording:
            recording = True; buffer = []
            print("[REC] Εγγραφη...")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    print(f"\n{'='*55}")
    print(f"Cloudflare: cloudflared tunnel --url http://localhost:{PORT}")
    print(f"Local:      http://localhost:{PORT}")
    print(f"{'='*55}\n")

    uvicorn.run(app, host=HOST, port=PORT)