from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
from pymongo import MongoClient
import numpy as np
import base64, os, tempfile, cv2
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)


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
    doc = get_col().find_one({"email": email}, {"embedding": 1})
    if doc and "embedding" in doc:
        return np.array(doc["embedding"], dtype=np.float64)
    return None

def db_save(email: str, embedding: np.ndarray):
    get_col().update_one(
        {"email": email},
        {"$set": {"email": email, "embedding": embedding.tolist()}},
        upsert=True,
    )

def db_count() -> int:
    try:
        return get_col().count_documents({})
    except Exception:
        return -1

# ── OPENCV FACE DETECTOR ──────────────────────────────────
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Loading DeepFace models...")
DeepFace.build_model("Facenet")
DeepFace.build_model("Emotion")
print("✅ Models ready!")

# ── HELPERS ───────────────────────────────────────────────
def b64_to_array(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",")[1]
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img)

def detect_and_crop(img_array: np.ndarray):
    gray      = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    face_rect = None
    for scale, neighbors, min_size in [
        (1.05, 3, (20, 20)),
        (1.08, 4, (40, 40)),
        (1.10, 5, (60, 60)),
    ]:
        faces = detector.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors,
            minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) > 0:
            face_rect = max(faces, key=lambda f: f[2] * f[3])
            print(f"  ✅ Face detected: {face_rect[2]}x{face_rect[3]} px")
            break
    if face_rect is None:
        print("  ⚠️ No face detected")
        return None
    x, y, w, h = face_rect
    pad_x = int(w * 0.20); pad_y = int(h * 0.20)
    x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
    x2 = min(img_array.shape[1], x + w + pad_x)
    y2 = min(img_array.shape[0], y + h + pad_y)
    crop    = img_array[y1:y2, x1:x2]
    lab     = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    crop    = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)
    crop    = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
    return crop

def save_temp(img_array: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    Image.fromarray(img_array).save(tmp.name, quality=95)
    tmp.close()
    return tmp.name

def get_embedding(img_path: str):
    try:
        result = DeepFace.represent(
            img_path=img_path, model_name="Facenet",
            enforce_detection=False, detector_backend="skip",
        )
        return np.array(result[0]["embedding"], dtype=np.float64)
    except Exception as e:
        print(f"  ⚠️ Embedding error: {e}")
        return None

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(1.0 - np.dot(a, b) / denom) if denom > 1e-10 else 1.0

def to_float(val) -> float:
    return round(float(val.item() if hasattr(val, "item") else val), 4)

def smart_emotion(raw: dict) -> str:
    e        = {k: to_float(v) for k, v in raw.items()}
    happy    = e.get("happy",    0.0)
    sad      = e.get("sad",      0.0)
    angry    = e.get("angry",    0.0)
    fear     = e.get("fear",     0.0)
    surprise = e.get("surprise", 0.0)
    disgust  = e.get("disgust",  0.0)
    neutral  = e.get("neutral",  100.0)

    print(f"  😊 happy={happy} sad={sad} angry={angry} fear={fear} surprise={surprise} neutral={neutral}")

    # Tier 1 — strong >= 30%
    if happy    >= 30.0: return "happy"
    if sad      >= 30.0: return "sad"
    if angry    >= 30.0: return "angry"
    if fear     >= 30.0: return "fear"
    if surprise >= 30.0: return "surprise"
    if disgust  >= 30.0: return "disgust"

    # Tier 2 — moderate >= 5% when not overwhelmingly neutral
    if neutral < 95.0:
        if happy    >= 5.0: return "happy"
        if sad      >= 5.0: return "sad"
        if angry    >= 5.0: return "angry"
        if fear     >= 5.0: return "fear"
        if surprise >= 5.0: return "surprise"
        if disgust  >= 1.0: return "disgust"

    # Tier 3 — subtle dominant
    non_neutral = {"happy": happy, "sad": sad, "angry": angry,
                   "fear": fear, "surprise": surprise, "disgust": disgust}
    top  = max(non_neutral, key=lambda k: non_neutral[k])
    rest = sum(v for k, v in non_neutral.items() if k != top)
    if non_neutral[top] >= 2.0 and non_neutral[top] > rest * 3:
        print(f"  → Subtle: {top}")
        return top

    return "neutral"

# ── ROUTES ────────────────────────────────────────────────

@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify({
        "status":       "VitaAI Face Server running ✅",
        "storage":      "MongoDB Atlas",
        "faces_stored": db_count(),
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
        img_array = b64_to_array(img_b64)
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Invalid image: {e}"}), 400

    face = detect_and_crop(img_array)
    if face is None:
        return jsonify({"ok": False, "msg": "No face detected. Come closer, look straight at camera, ensure good lighting."})

    tmp = save_temp(face)
    try:
        emb = get_embedding(tmp)
        if emb is None:
            return jsonify({"ok": False, "msg": "Could not process face. Please try again."})
        db_save(email, emb)
        print(f"  ✅ Registered: {email} | Total: {db_count()}")
        return jsonify({"ok": True, "msg": "Face registered successfully!"})
    except Exception as e:
        print(f"  ❌ DB error: {e}")
        return jsonify({"ok": False, "msg": "Database error. Please try again."}), 500
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


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

    stored_emb = db_load(email)
    if stored_emb is None:
        return jsonify({"ok": False, "msg": "Face not registered. Please sign up first."})

    try:
        img_array = b64_to_array(img_b64)
    except Exception as e:
        return jsonify({"ok": False, "msg": f"Invalid image: {e}"}), 400

    face = detect_and_crop(img_array)
    if face is None:
        return jsonify({"ok": False, "msg": "No face detected. Come closer, look straight at camera, ensure good lighting."})

    tmp = save_temp(face)
    try:
        emb = get_embedding(tmp)
        if emb is None:
            return jsonify({"ok": False, "msg": "Could not process face. Please try again."})
        dist = cosine_distance(stored_emb, emb)
        ok   = dist < 0.55
        print(f"  📊 Distance: {dist:.4f} | {'✅ PASS' if ok else '❌ FAIL'}")
        return jsonify({
            "ok":  bool(ok),
            "msg": "Face verified! Welcome back 👋" if ok else "Face did not match. Please try again.",
        })
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


@app.route("/face/emotion", methods=["POST"])
def face_emotion():
    print("\n😊 EMOTION")
    d       = request.get_json(silent=True) or {}
    img_b64 = d.get("img", "")

    if not img_b64:
        return jsonify({"ok": False, "emotion": "neutral", "msg": "Image is required."}), 400

    try:
        img_array = b64_to_array(img_b64)
    except Exception as e:
        return jsonify({"ok": False, "emotion": "neutral", "msg": f"Invalid image: {e}"}), 400

    face = detect_and_crop(img_array)
    if face is None:
        return jsonify({"ok": False, "emotion": "neutral", "msg": "No face detected. Move closer to the camera."})

    tmp = save_temp(face)
    try:
        result = DeepFace.analyze(
            img_path=tmp, actions=["emotion"],
            enforce_detection=False, detector_backend="skip", silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        raw     = result.get("emotion", {})
        clean   = {k: to_float(v) for k, v in raw.items()}
        emotion = smart_emotion(raw)
        print(f"  🏆 Final: {emotion}")
        return jsonify({
            "ok":      True,
            "emotion": emotion,
            "scores": {
                "happy":    clean.get("happy",    0),
                "sad":      clean.get("sad",      0),
                "angry":    clean.get("angry",    0),
                "fear":     clean.get("fear",     0),
                "surprise": clean.get("surprise", 0),
                "disgust":  clean.get("disgust",  0),
                "neutral":  clean.get("neutral",  0),
            },
        })
    except Exception as e:
        print(f"  ⚠️ Emotion error: {e}")
        return jsonify({"ok": False, "emotion": "neutral", "msg": str(e)})
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


# ── ENTRYPOINT ────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print(f"  VitaAI Face Server  —  port {port}")
    print(f"  MongoDB faces: {db_count()}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)