"""
VitaAI Face Server — Simple & Reliable
- Face Detection    → OpenCV Haar Cascade
- Face Identity     → OpenCV ORB feature matching (no dlib, no DeepFace)
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
    doc = get_col().find_one({"email": email}, {"descriptors": 1})
    if doc and "descriptors" in doc:
        return [np.array(d, dtype=np.uint8) for d in doc["descriptors"]]
    return None

def db_save(email: str, descriptor: np.ndarray):
    """Store up to 3 descriptors per user for robustness."""
    doc = get_col().find_one({"email": email}, {"descriptors": 1})
    if doc and "descriptors" in doc:
        existing = doc["descriptors"]
        if len(existing) >= 3:
            existing = existing[-2:]
        existing.append(descriptor.tolist())
        get_col().update_one({"email": email}, {"$set": {"descriptors": existing}})
    else:
        get_col().update_one(
            {"email": email},
            {"$set": {"email": email, "descriptors": [descriptor.tolist()]}},
            upsert=True,
        )

def db_count() -> int:
    try:
        return get_col().count_documents({})
    except Exception:
        return -1

# ── OPENCV MODELS ─────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
orb = cv2.ORB_create(nfeatures=500)

# FER emotion detector
emotion_detector = FER(mtcnn=False)

print("✅ OpenCV face detector ready!")
print("✅ ORB feature extractor ready!")
print("✅ FER emotion detector ready!")

# ── IMAGE HELPERS ─────────────────────────────────────────
def b64_to_bgr(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",")[1]
    img_rgb = np.array(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def detect_face_region(img_bgr: np.ndarray):
    """Detect and return cropped face as grayscale 128x128."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Try multiple detection passes
    for scale, neighbors in [(1.1, 5), (1.05, 3), (1.05, 2)]:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(30, 30)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            print(f"  ✅ Face: {w}x{h} at ({x},{y})")
            # Crop with padding
            pad = int(min(w, h) * 0.15)
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(img_bgr.shape[1], x + w + pad)
            y2 = min(img_bgr.shape[0], y + h + pad)
            face = gray[y1:y2, x1:x2]
            face = cv2.resize(face, (128, 128))
            face = cv2.equalizeHist(face)
            return face

    print("  ⚠️ No face detected")
    return None

def get_orb_descriptor(face_gray: np.ndarray):
    """Extract ORB feature descriptor from face image."""
    keypoints, descriptors = orb.detectAndCompute(face_gray, None)
    if descriptors is None or len(keypoints) < 5:
        print("  ⚠️ Not enough keypoints")
        return None
    print(f"  ✅ {len(keypoints)} keypoints found")
    # Take first 200 descriptors for consistent size
    return descriptors[:200]

def match_descriptors(stored_list: list, live_desc: np.ndarray) -> float:
    """
    Match live descriptor against all stored descriptors.
    Returns best match score (0-100, higher = better match).
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_score = 0.0

    for stored_desc in stored_list:
        stored = np.array(stored_desc, dtype=np.uint8)
        try:
            matches = bf.match(stored, live_desc)
            if not matches:
                continue
            # Sort by distance — lower distance = better match
            matches = sorted(matches, key=lambda x: x.distance)
            # Take top 50 matches
            good = matches[:50]
            # Score: average distance (lower = better) → convert to 0-100
            avg_dist = np.mean([m.distance for m in good])
            score = max(0, 100 - avg_dist)
            print(f"  📊 Match score: {score:.1f} (avg_dist={avg_dist:.1f})")
            best_score = max(best_score, score)
        except Exception as e:
            print(f"  ⚠️ Match error: {e}")
            continue

    return best_score

# ── FER EMOTION ───────────────────────────────────────────
def detect_emotion(img_bgr: np.ndarray):
    try:
        results = emotion_detector.detect_emotions(img_bgr)
        if not results:
            return "neutral", {}
        best     = max(results, key=lambda r: r["box"][2] * r["box"][3])
        emotions = best["emotions"]
        dominant = max(emotions, key=emotions.get)
        score    = emotions[dominant]
        print(f"  😊 {emotions}")
        print(f"  🏆 {dominant} ({score:.3f})")
        if dominant == "neutral" and score < 0.70:
            others = {k: v for k, v in emotions.items() if k != "neutral"}
            if others:
                second = max(others, key=others.get)
                if emotions[second] >= 0.15:
                    return second, emotions
        return dominant, emotions
    except Exception as e:
        print(f"  ⚠️ FER error: {e}")
        return "neutral", {}

# ── ROUTES ────────────────────────────────────────────────

@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify({
        "status":       "VitaAI Face Server ✅",
        "face_detect":  "OpenCV Haar Cascade",
        "identity":     "OpenCV ORB Feature Matching",
        "emotion":      "FER library",
        "storage":      "MongoDB Atlas",
        "faces_stored": db_count(),
    }), 200


@app.route("/face/register", methods=["POST"])
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

    face = detect_face_region(img_bgr)
    if face is None:
        return jsonify({"ok": False, "msg": "No face detected. Come closer & look straight at camera."})

    desc = get_orb_descriptor(face)
    if desc is None:
        return jsonify({"ok": False, "msg": "Could not read face features. Ensure good lighting."})

    try:
        db_save(email, desc)
        doc   = get_col().find_one({"email": email}, {"descriptors": 1})
        count = len(doc["descriptors"]) if doc else 1
        print(f"  ✅ {email} — {count} descriptor(s) stored")
        msg = "Face registered!" if count == 1 else f"Face updated! ({count}/3 angles stored)"
        return jsonify({"ok": True, "msg": msg})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Database error: {e}"}), 500


@app.route("/face/verify", methods=["POST"])
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

    face = detect_face_region(img_bgr)
    if face is None:
        return jsonify({"ok": False, "msg": "No face detected. Come closer & look straight at camera."})

    desc = get_orb_descriptor(face)
    if desc is None:
        return jsonify({"ok": False, "msg": "Could not read face features. Ensure good lighting."})

    score = match_descriptors(stored, desc)
    ok    = score >= 40.0  # threshold: score >= 40 = same person

    print(f"  {'✅ PASS' if ok else '❌ FAIL'} (score={score:.1f}, threshold=40)")

    return jsonify({
        "ok":  bool(ok),
        "msg": "Face verified! Welcome back 👋" if ok else "Face did not match. Please try again.",
    })


@app.route("/face/emotion", methods=["POST"])
def face_emotion():
    print("\n😊 EMOTION")
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
    print(f"  Identity : OpenCV ORB")
    print(f"  Emotion  : FER library")
    print(f"  Storage  : MongoDB Atlas")
    print(f"  Faces    : {db_count()}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)