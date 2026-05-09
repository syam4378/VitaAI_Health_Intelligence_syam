from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import numpy as np
import base64, os, cv2
from PIL import Image
from io import BytesIO
from datetime import datetime
import onnxruntime as ort

app = Flask(__name__)
CORS(app, origins=["*"])

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD"
    return response

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        from flask import make_response
        r = make_response("", 200)
        r.headers["Access-Control-Allow-Origin"]  = "*"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD"
        return r

# ── CONFIG ────────────────────────────────────────────────
MODELS_DIR   = "/app/models"
ARCFACE_PATH = f"{MODELS_DIR}/arcface_w600k_r50.onnx"
EMOTION_PATH = f"{MODELS_DIR}/emotion_miniXception.onnx"
MODEL_VERSION = "haar+arcface+v2"

# Very relaxed — easy signup/login
SIMILARITY_THRESHOLD = 0.35
MIN_FACE_SIZE        = 60    # px
MODEL_VERSION        = "haar+arcface+v2"

# FERPlus-8 labels
EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]

# ArcFace 5-point reference
ARCFACE_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# ── MONGODB ───────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "")
_client = None
_col    = None

def get_col():
    global _client, _col
    if _col is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _col    = _client["vitaai"]["faces"]
        _col.create_index("email", unique=True)
    return _col

def db_load(email):
    doc = get_col().find_one({"email": email}, {"embeddings": 1})
    if doc and "embeddings" in doc:
        return [np.array(e["vec"], dtype=np.float32) for e in doc["embeddings"]]
    return None

def db_save(email, embedding):
    doc = get_col().find_one({"email": email}, {"embeddings": 1})
    entry = {"vec": embedding.tolist(), "ts": datetime.utcnow().isoformat(), "model": MODEL_VERSION}
    if doc and "embeddings" in doc:
        existing = doc["embeddings"]
        if len(existing) >= 5:
            existing = existing[-4:]
        existing.append(entry)
        get_col().update_one({"email": email}, {"$set": {"embeddings": existing}})
    else:
        get_col().update_one(
            {"email": email},
            {"$set": {"email": email, "embeddings": [entry]}},
            upsert=True
        )

def db_count():
    try:    return get_col().count_documents({})
    except: return -1

# ── LOAD MODELS ───────────────────────────────────────────
arcface_sess = None
emotion_sess = None

def load_models():
    global arcface_sess, emotion_sess
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2

    if os.path.exists(ARCFACE_PATH):
        arcface_sess = ort.InferenceSession(ARCFACE_PATH, opts, providers=["CPUExecutionProvider"])
        print("✅ ArcFace loaded")
    else:
        print("❌ ArcFace missing!")

    if os.path.exists(EMOTION_PATH):
        emotion_sess = ort.InferenceSession(EMOTION_PATH, opts, providers=["CPUExecutionProvider"])
        print("✅ Emotion FERPlus-8 loaded")
    else:
        print("⚠️  Emotion model missing")

load_models()

# ── HAAR CASCADE (reliable, always works) ─────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ── IMAGE HELPERS ─────────────────────────────────────────
def b64_to_bgr(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    arr = np.array(img)
    # Resize if too large (speed)
    h, w = arr.shape[:2]
    if w > 640:
        scale = 640 / w
        arr = cv2.resize(arr, (640, int(h * scale)))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# ── FACE DETECTION ────────────────────────────────────────
def detect_face(img_bgr):
    """
    Detect ONE face. Tries multiple scaleFactor values for reliability.
    Returns (x,y,w,h) or None.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Equalise histogram for better detection in dark/bright conditions
    gray = cv2.equalizeHist(gray)

    # Try progressively more relaxed settings
    configs = [
        (1.1, 5, (80, 80)),
        (1.1, 3, (60, 60)),
        (1.05, 3, (50, 50)),
        (1.05, 2, (40, 40)),
        (1.03, 2, (30, 30)),
    ]
    for sf, mn, ms in configs:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=ms)
        if len(faces) > 0:
            # Return largest face
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            return faces[0]
    return None

# ── FACE ALIGNMENT ────────────────────────────────────────
def align_or_crop(img_bgr, x, y, w, h):
    """
    Try to align using eye positions. Fall back to simple crop + resize.
    Always returns a 112x112 image.
    """
    # Add margin around face
    margin = int(max(w, h) * 0.2)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_bgr.shape[1], x + w + margin)
    y2 = min(img_bgr.shape[0], y + h + margin)
    face_crop = img_bgr[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    # Try eye detection for alignment
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.equalizeHist(gray_face)
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))

    if len(eyes) >= 2:
        # Sort eyes left to right
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]
        eye1_center = (ex1 + ew1//2, ey1 + eh1//2)
        eye2_center = (ex2 + ew2//2, ey2 + eh2//2)

        # Compute rotation angle
        dy = eye2_center[1] - eye1_center[1]
        dx = eye2_center[0] - eye1_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate to align eyes horizontally
        center = (face_crop.shape[1]//2, face_crop.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(face_crop, M, (face_crop.shape[1], face_crop.shape[0]))
        return cv2.resize(aligned, (112, 112))

    # No eyes found — just resize crop
    return cv2.resize(face_crop, (112, 112))

# ── ARCFACE EMBEDDING ─────────────────────────────────────
def get_embedding(face_112):
    face_rgb  = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB).astype(np.float32)
    face_norm = (face_rgb - 127.5) / 128.0
    face_inp  = face_norm.transpose(2, 0, 1)[np.newaxis]
    inp_name  = arcface_sess.get_inputs()[0].name
    emb       = arcface_sess.run(None, {inp_name: face_inp})[0][0]
    norm      = np.linalg.norm(emb)
    return (emb / (norm + 1e-10)).astype(np.float32)

def cosine_sim(a, b):
    return float(np.dot(a, b))

def best_sim(stored, live):
    sims = [cosine_sim(s, live) for s in stored]
    best = max(sims)
    print(f"  📊 Sims: {[round(s,4) for s in sims]} | Best: {best:.4f}")
    return best

# ── EMOTION DETECTION ─────────────────────────────────────
def detect_emotion(img_bgr):
    face = detect_face(img_bgr)
    if face is None:
        return "neutral", {}

    x, y, w, h = face
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    roi  = gray[y:y+h, x:x+w]
    if roi.size == 0:
        return "neutral", {}

    if emotion_sess is not None:
        try:
            # FERPlus-8: (1,1,64,64), raw 0-255 float
            face_64  = cv2.resize(roi, (64, 64)).astype(np.float32)
            face_inp = face_64[np.newaxis, np.newaxis]
            inp_name = emotion_sess.get_inputs()[0].name
            preds    = emotion_sess.run(None, {inp_name: face_inp})[0][0]

            # Softmax
            preds = np.exp(preds - preds.max())
            preds /= preds.sum()
            scores = {EMOTION_LABELS[i]: float(preds[i]) for i in range(len(EMOTION_LABELS))}

            # ── FIX: proper dominant emotion selection ──
            # Sort by score descending
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            dominant, top_score = sorted_scores[0]
            second,  sec_score  = sorted_scores[1]

            print(f"  😊 Top: {dominant}={top_score:.3f}, 2nd: {second}={sec_score:.3f}")

            # Only return neutral if it clearly wins (>55%)
            # Otherwise return the strongest non-neutral emotion
            if dominant == "neutral" and top_score < 0.55:
                # Promote second if meaningful
                if sec_score >= 0.15:
                    print(f"  → Promoting {second} over weak neutral")
                    return second, scores

            # Map contempt→disgust for UI (contempt not in moodMap)
            if dominant == "contempt":
                dominant = "disgust"

            return dominant, scores

        except Exception as e:
            print(f"  ⚠️  Emotion inference error: {e}")

    # Fallback: pixel-based
    var    = float(np.var(roi))
    bright = float(np.mean(roi))
    scores = {l: 0.0 for l in EMOTION_LABELS}
    if var > 800 and bright > 90:
        scores["happy"] = 1.0; return "happy", scores
    elif bright < 60:
        scores["sad"] = 1.0; return "sad", scores
    scores["neutral"] = 1.0
    return "neutral", scores

# ── FACE PIPELINE ─────────────────────────────────────────
def face_pipeline(img_bgr):
    if arcface_sess is None:
        return None, "Face recognition model not loaded."

    face = detect_face(img_bgr)
    if face is None:
        return None, "No face detected. Look straight at camera in good light."

    x, y, w, h = face
    print(f"  ✅ Face: ({x},{y},{w},{h})")

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None, f"Face too small ({w}x{h}px). Come closer to camera."

    aligned = align_or_crop(img_bgr, x, y, w, h)
    if aligned is None:
        return None, "Could not process face. Try again."

    emb = get_embedding(aligned)
    return emb, None

# ── ROUTES ────────────────────────────────────────────────
@app.route("/", methods=["GET", "HEAD", "OPTIONS"])
def index():
    return jsonify({
        "status":       "VitaAI Face Server ✅",
        "detection":    "OpenCV Haar (reliable)",
        "recognition":  "ArcFace ONNX 512-D" if arcface_sess else "❌ NOT LOADED",
        "emotion":      "FERPlus-8 ONNX"      if emotion_sess else "fallback",
        "threshold":    SIMILARITY_THRESHOLD,
        "storage":      "MongoDB Atlas",
        "faces_stored": db_count(),
    }), 200


@app.route("/face/register", methods=["POST", "OPTIONS"])
def face_register():
    d       = request.get_json(silent=True) or {}
    email   = d.get("email", "").strip()
    img_b64 = d.get("img", "")
    print(f"\n📝 REGISTER: {email!r}")

    if not email or not img_b64:
        return jsonify({"ok": False, "msg": "Email and image required."}), 400
    try:
        img_bgr = b64_to_bgr(img_b64)
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Invalid image: {e}"}), 400

    emb, err = face_pipeline(img_bgr)
    if err:
        return jsonify({"ok": False, "msg": err})

    try:
        db_save(email, emb)
        doc   = get_col().find_one({"email": email}, {"embeddings": 1})
        count = len(doc["embeddings"]) if doc else 1
        msg   = "Face registered! ✅" if count == 1 else f"Face updated! ({count}/5 angles stored)"
        print(f"  ✅ Saved — {count} embedding(s)")
        return jsonify({"ok": True, "msg": msg, "count": count})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Database error: {e}"}), 500


@app.route("/face/verify", methods=["POST", "OPTIONS"])
def face_verify():
    d       = request.get_json(silent=True) or {}
    email   = d.get("email", "").strip()
    img_b64 = d.get("img", "")
    print(f"\n🔍 VERIFY: {email!r}")

    if not email or not img_b64:
        return jsonify({"ok": False, "msg": "Email and image required."}), 400

    stored = db_load(email)
    if stored is None:
        return jsonify({"ok": False, "msg": "Face not registered. Please sign up first."})

    try:
        img_bgr = b64_to_bgr(img_b64)
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Invalid image: {e}"}), 400

    live_emb, err = face_pipeline(img_bgr)
    if err:
        return jsonify({"ok": False, "msg": err})

    sim = best_sim(stored, live_emb)
    ok  = sim >= SIMILARITY_THRESHOLD
    print(f"  {'✅ PASS' if ok else '❌ FAIL'} sim={sim:.4f} threshold={SIMILARITY_THRESHOLD}")

    return jsonify({
        "ok":  bool(ok),
        "msg": "Face verified! Welcome back 👋" if ok else f"Face did not match (score: {sim:.2f}). Try again or register your face again.",
    })


@app.route("/face/emotion", methods=["POST", "OPTIONS"])
def face_emotion():
    print("\n😊 EMOTION")
    d       = request.get_json(silent=True) or {}
    img_b64 = d.get("img", "")

    if not img_b64:
        return jsonify({"ok": False, "emotion": "neutral"}), 400
    try:
        img_bgr = b64_to_bgr(img_b64)
    except Exception as e:
        return jsonify({"ok": False, "emotion": "neutral", "msg": str(e)}), 400

    emotion, scores = detect_emotion(img_bgr)
    print(f"  🏆 Final: {emotion}")

    return jsonify({
        "ok":      True,
        "emotion": emotion,
        "scores":  {k: round(float(v), 4) for k, v in scores.items()},
    })


# ── ENTRYPOINT ────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("=" * 55)
    print(f"  VitaAI Face Server — port {port}")
    print(f"  Detection   : OpenCV Haar (reliable)")
    print(f"  Recognition : {'ArcFace ONNX 512-D' if arcface_sess else '❌ NOT LOADED'}")
    print(f"  Emotion     : {'FERPlus-8 ONNX' if emotion_sess else 'fallback'}")
    print(f"  Threshold   : {SIMILARITY_THRESHOLD}")
    print(f"  Faces stored: {db_count()}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False)