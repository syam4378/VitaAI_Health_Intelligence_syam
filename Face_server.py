"""
VitaAI Face Server — Robust & Practical
- Face Detection    → dlib HOG (reliable for frontal webcam faces)
- Face Identity     → dlib 128D ResNet + stores 3 encodings per user
- Emotion Detection → FER library
- Storage           → MongoDB Atlas
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from fer import FER
from pymongo import MongoClient
import numpy as np
import base64, os, cv2
from PIL import Image
from io import BytesIO
import dlib

app = Flask(__name__)
CORS(app)

# ── MONGODB ───────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "")
_client   = None
_col      = None

def get_col():
    global _client, _col
    if _col is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _col    = _client["vitaai"]["faces"]
    return _col

def db_load(email: str):
    """Load all stored encodings for a user (up to 3)."""
    doc = get_col().find_one({"email": email}, {"encodings": 1})
    if doc and "encodings" in doc:
        return [np.array(e, dtype=np.float64) for e in doc["encodings"]]
    return None

def db_save_encoding(email: str, encoding: np.ndarray):
    """
    Store up to 3 encodings per user.
    Each registration adds one — up to max 3.
    This makes verification robust across lighting/angle changes.
    """
    doc = get_col().find_one({"email": email}, {"encodings": 1})
    if doc and "encodings" in doc:
        existing = doc["encodings"]
        if len(existing) >= 3:
            # Replace oldest encoding (keep last 2 + new one)
            existing = existing[-2:]
        existing.append(encoding.tolist())
        get_col().update_one(
            {"email": email},
            {"$set": {"encodings": existing}}
        )
    else:
        get_col().update_one(
            {"email": email},
            {"$set": {"email": email, "encodings": [encoding.tolist()]}},
            upsert=True,
        )

def db_count() -> int:
    try:
        return get_col().count_documents({})
    except Exception:
        return -1

# ── LOAD DLIB MODELS ──────────────────────────────────────
DLIB_PREDICTOR = "/app/models/shape_predictor_68_face_landmarks.dat"
DLIB_FACE_REC  = "/app/models/dlib_face_recognition_resnet_model_v1.dat"

print("Loading dlib models...")
hog_detector = dlib.get_frontal_face_detector()
predictor    = dlib.shape_predictor(DLIB_PREDICTOR)
face_rec     = dlib.face_recognition_model_v1(DLIB_FACE_REC)
print("✅ dlib models ready!")

emotion_detector = FER(mtcnn=False)
print("✅ FER emotion detector ready!")

# ── IMAGE HELPERS ─────────────────────────────────────────
def b64_to_rgb(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",")[1]
    return np.array(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))

def b64_to_bgr(b64: str) -> np.ndarray:
    return cv2.cvtColor(b64_to_rgb(b64), cv2.COLOR_RGB2BGR)

# ── DLIB FACE ENCODING ────────────────────────────────────
def get_face_encoding(img_rgb: np.ndarray):
    """
    Detect face → get 128D encoding.
    Tries upsample=1 first (better for small/distant faces),
    falls back to upsample=0 for speed if face is large.
    """
    # Resize if too large (speeds up detection)
    h, w = img_rgb.shape[:2]
    if w > 640:
        scale   = 640 / w
        img_rgb = cv2.resize(img_rgb, (640, int(h * scale)))

    # Try with upsampling first (catches smaller/further faces)
    dets = hog_detector(img_rgb, 1)

    if len(dets) == 0:
        # Try without upsampling (faster, for close faces)
        dets = hog_detector(img_rgb, 0)

    if len(dets) == 0:
        print("  ⚠️ No face detected")
        return None

    # Pick largest face
    det   = max(dets, key=lambda d: (d.right()-d.left()) * (d.bottom()-d.top()))
    pw    = det.right() - det.left()
    ph    = det.bottom() - det.top()
    print(f"  ✅ Face detected: {pw}x{ph} px")

    # Get 68 landmarks + compute 128D encoding
    shape    = predictor(img_rgb, det)
    encoding = face_rec.compute_face_descriptor(img_rgb, shape, num_jitters=2)
    # num_jitters=2: slightly jitters image for more stable encoding
    return np.array(encoding, dtype=np.float64)

def best_distance(stored_encodings: list, live_enc: np.ndarray) -> float:
    """
    Compare live encoding against ALL stored encodings.
    Return the BEST (smallest) distance.
    This is how robust multi-encoding systems work —
    if ANY stored encoding matches, the person is verified.
    """
    distances = [np.linalg.norm(s - live_enc) for s in stored_encodings]
    best = min(distances)
    print(f"  📊 All distances: {[round(d, 4) for d in distances]}")
    print(f"  📊 Best distance: {best:.4f}")
    return best

# ── FER EMOTION DETECTION ─────────────────────────────────
def detect_emotion(img_bgr: np.ndarray):
    """FER emotion detection — same approach as emotionDetection.py from zip."""
    try:
        results = emotion_detector.detect_emotions(img_bgr)

        if not results:
            print("  ⚠️ FER: No emotions detected")
            return "neutral", {}

        best     = max(results, key=lambda r: r["box"][2] * r["box"][3])
        emotions = best["emotions"]
        dominant = max(emotions, key=emotions.get)
        score    = emotions[dominant]

        print(f"  😊 FER: {emotions}")
        print(f"  🏆 Dominant: {dominant} ({score:.3f})")

        # If neutral but not very confident, check for subtle emotion
        if dominant == "neutral" and score < 0.70:
            others = {k: v for k, v in emotions.items() if k != "neutral"}
            if others:
                second = max(others, key=others.get)
                if emotions[second] >= 0.15:
                    print(f"  → Override: {second} ({emotions[second]:.3f})")
                    return second, emotions

        return dominant, emotions

    except Exception as e:
        print(f"  ⚠️ FER error: {e}")
        return "neutral", {}

# ── ROUTES ────────────────────────────────────────────────

@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify({
        "status":        "VitaAI Face Server ✅",
        "face_detect":   "dlib HOG",
        "identity":      "dlib 128D ResNet (3 encodings/user)",
        "emotion":       "FER library",
        "storage":       "MongoDB Atlas",
        "faces_stored":  db_count(),
    }), 200


@app.route("/face/register", methods=["POST"])
def face_register():
    d       = request.get_json(silent=True) or {}
    email   = d.get("email", "").strip()
    img_b64 = d.get("img", "")
    print(f"\n📝 REGISTER: {email!r}")

    if not email:
        return jsonify({"ok": False, "msg": "Email is required."}), 400
    if not img_b64:
        return jsonify({"ok": False, "msg": "Image is required."}), 400

    try:
        img_rgb = b64_to_rgb(img_b64)
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Invalid image: {e}"}), 400

    encoding = get_face_encoding(img_rgb)
    if encoding is None:
        return jsonify({
            "ok":  False,
            "msg": "No face detected. Come closer, look straight at camera, ensure good lighting."
        })

    try:
        db_save_encoding(email, encoding)

        # Tell user how many encodings stored
        doc   = get_col().find_one({"email": email}, {"encodings": 1})
        count = len(doc["encodings"]) if doc else 1

        msg = "Face registered!" if count == 1 else f"Face updated ({count}/3 angles stored — more = better accuracy!)"
        print(f"  ✅ {email} has {count} encodings stored")
        return jsonify({"ok": True, "msg": msg, "encodings_stored": count})
    except Exception as e:
        print(f"  ❌ DB error: {e}")
        return jsonify({"ok": False, "msg": "Database error. Please try again."}), 500


@app.route("/face/verify", methods=["POST"])
def face_verify():
    d       = request.get_json(silent=True) or {}
    email   = d.get("email", "").strip()
    img_b64 = d.get("img", "")
    print(f"\n🔍 VERIFY: {email!r}")

    if not email:
        return jsonify({"ok": False, "msg": "Email is required."}), 400
    if not img_b64:
        return jsonify({"ok": False, "msg": "Image is required."}), 400

    stored_encs = db_load(email)
    if stored_encs is None:
        return jsonify({"ok": False, "msg": "Face not registered. Please sign up first."})

    try:
        img_rgb = b64_to_rgb(img_b64)
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Invalid image: {e}"}), 400

    live_enc = get_face_encoding(img_rgb)
    if live_enc is None:
        return jsonify({
            "ok":  False,
            "msg": "No face detected. Come closer, look straight at camera."
        })

    dist = best_distance(stored_encs, live_enc)

    # Threshold: 0.50 strict — adjustable here
    # < 0.45 = very confident match
    # 0.45-0.50 = good match
    # > 0.50 = likely different person
    THRESHOLD = 0.50
    ok        = dist < THRESHOLD

    print(f"  {'✅ PASS' if ok else '❌ FAIL'} (threshold={THRESHOLD})")

    return jsonify({
        "ok":  bool(ok),
        "msg": "Face verified! Welcome back 👋" if ok else "Face did not match. Please try again.",
    })


@app.route("/face/emotion", methods=["POST"])
def face_emotion():
    print("\n😊 EMOTION DETECT")
    d       = request.get_json(silent=True) or {}
    img_b64 = d.get("img", "")

    if not img_b64:
        return jsonify({"ok": False, "emotion": "neutral", "msg": "Image required."}), 400

    try:
        img_bgr = b64_to_bgr(img_b64)
    except Exception as e:
        return jsonify({"ok": False, "emotion": "neutral", "msg": f"Invalid image: {e}"}), 400

    emotion, scores = detect_emotion(img_bgr)
    print(f"  🏆 Final: {emotion}")

    return jsonify({
        "ok":      True,
        "emotion": emotion,
        "scores": {
            "happy":    round(float(scores.get("happy",    0)), 4),
            "sad":      round(float(scores.get("sad",      0)), 4),
            "angry":    round(float(scores.get("angry",    0)), 4),
            "fear":     round(float(scores.get("fear",     0)), 4),
            "surprise": round(float(scores.get("surprise", 0)), 4),
            "disgust":  round(float(scores.get("disgust",  0)), 4),
            "neutral":  round(float(scores.get("neutral",  0)), 4),
        },
    })


# ── ENTRYPOINT ────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("=" * 50)
    print(f"  VitaAI Face Server  —  port {port}")
    print(f"  Identity    : dlib 128D (3 encodings/user)")
    print(f"  Emotion     : FER library")
    print(f"  Storage     : MongoDB Atlas")
    print(f"  Faces stored: {db_count()}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)