from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
import base64, os, pickle, tempfile, cv2
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)  # ← MUST be right after app, before all routes

# ── PERSISTENT STORAGE — survives Render redeploys ────────
# On Render: add a Disk with mount path /data
# Locally: saves in current folder

@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify({"status": "VitaAI Face Server running"}), 200
DATA_DIR = "/data" if os.path.exists("/data") else "."
DB = os.path.join(DATA_DIR, "faces.pkl")

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Loading Facenet model...")
DeepFace.build_model("Facenet")
print("✅ Model ready! Server starting...")

def load():
    return pickle.load(open(DB, "rb")) if os.path.exists(DB) else {}

def save(d):
    pickle.dump(d, open(DB, "wb"))

def b64_to_array(b64):
    data = base64.b64decode(b64.split(",")[1])
    img  = Image.open(BytesIO(data)).convert("RGB")
    return np.array(img)

def detect_and_crop_face(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    face_rect = None
    for scale, neighbors, min_size in [
        (1.05, 3, (30, 30)),
        (1.1,  4, (50, 50)),
        (1.1,  5, (80, 80)),
    ]:
        faces = detector.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=min_size)
        if len(faces) > 0:
            face_rect = max(faces, key=lambda f: f[2] * f[3])
            print(f"  ✅ Face detected {face_rect[2]}x{face_rect[3]} px")
            break
    if face_rect is None:
        print("  ⚠️ No face detected")
        return None
    x, y, w, h = face_rect
    pad_x = int(w * 0.20); pad_y = int(h * 0.20)
    x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
    x2 = min(img_array.shape[1], x + w + pad_x)
    y2 = min(img_array.shape[0], y + h + pad_y)
    face_crop = img_array[y1:y2, x1:x2]
    lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    face_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    face_crop = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_CUBIC)
    return face_crop

def save_temp(img_array):
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    Image.fromarray(img_array).save(tmp.name, quality=95)
    return tmp.name

def get_embedding(img_path):
    try:
        result = DeepFace.represent(
            img_path=img_path, model_name="Facenet",
            enforce_detection=False, detector_backend="skip"
        )
        return np.array(result[0]["embedding"])
    except Exception as e:
        print(f"  ⚠️ Embedding error: {e}")
        return None

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def to_float(val):
    if hasattr(val, 'item'):
        return round(float(val.item()), 4)
    return round(float(val), 4)

def smart_emotion(emotions_raw: dict) -> str:
    e = {k: to_float(v) for k, v in emotions_raw.items()}
    neutral  = e.get('neutral',  100.0)
    happy    = e.get('happy',    0.0)
    sad      = e.get('sad',      0.0)
    angry    = e.get('angry',    0.0)
    fear     = e.get('fear',     0.0)
    surprise = e.get('surprise', 0.0)
    disgust  = e.get('disgust',  0.0)

    print(f"  📊 happy={happy}% sad={sad}% angry={angry}% fear={fear}% surprise={surprise}% neutral={neutral}%")

    # Strong emotion > 30% → always pick it
    if happy    >= 30.0: return 'happy'
    if sad      >= 30.0: return 'sad'
    if angry    >= 30.0: return 'angry'
    if fear     >= 30.0: return 'fear'
    if surprise >= 30.0: return 'surprise'

    # Moderate emotion 5-30% with neutral < 95%
    if neutral < 95.0:
        if sad      >= 5.0: return 'sad'
        if happy    >= 5.0: return 'happy'
        if angry    >= 5.0: return 'angry'
        if fear     >= 5.0: return 'fear'
        if surprise >= 5.0: return 'surprise'
        if disgust  >= 1.0: return 'disgust'

    # Subtle dominant emotion
    non_neutral = {'happy': happy, 'sad': sad, 'angry': angry,
                   'fear': fear, 'surprise': surprise, 'disgust': disgust}
    top_emotion = max(non_neutral, key=lambda k: non_neutral[k])
    top_score   = non_neutral[top_emotion]
    others_sum  = sum(v for k, v in non_neutral.items() if k != top_emotion)
    if top_score >= 2.0 and top_score > (others_sum * 3):
        print(f"  → Subtle: {top_emotion} ({top_score}%)")
        return top_emotion

    return 'neutral'


# ── ROUTES ────────────────────────────────────────────────

@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify({"status": "VitaAI Face Server running", "db": DB}), 200


@app.route("/face/register", methods=["POST"])
def face_register():
    d     = request.json
    email = d.get("email", "unknown")
    print(f"\n📝 REGISTER: {email}")
    img_array = b64_to_array(d["img"])
    face      = detect_and_crop_face(img_array)
    if face is None:
        return jsonify({"ok": False, "msg": "No face detected. Come closer & look straight at camera."})
    tmp = save_temp(face)
    try:
        embedding = get_embedding(tmp)
        if embedding is None:
            return jsonify({"ok": False, "msg": "Could not process face. Please try again."})
        db = load()
        db[email] = embedding
        save(db)
        print(f"  ✅ Registered: {email} | Total faces: {len(db)}")
        return jsonify({"ok": True, "msg": "Face registered!"})
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


@app.route("/face/verify", methods=["POST"])
def face_verify():
    d     = request.json
    email = d.get("email", "unknown")
    db    = load()
    print(f"\n🔍 VERIFY: {email} | Registered: {list(db.keys())}")
    if email not in db:
        return jsonify({"ok": False, "msg": "Face not registered. Please signup first."})
    img_array = b64_to_array(d["img"])
    face      = detect_and_crop_face(img_array)
    if face is None:
        return jsonify({"ok": False, "msg": "No face detected. Come closer & look straight at camera."})
    tmp = save_temp(face)
    try:
        embedding = get_embedding(tmp)
        if embedding is None:
            return jsonify({"ok": False, "msg": "Could not process face. Please try again."})
        dist = cosine_distance(db[email], embedding)
        ok   = dist < 0.6
        print(f"  Distance: {dist:.4f} | {'✅ PASS' if ok else '❌ FAIL'}")
        return jsonify({"ok": bool(ok), "msg": "Face verified!" if ok else "Face not matched. Try again."})
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


@app.route("/face/emotion", methods=["POST"])
def face_emotion():
    print(f"\n😊 EMOTION DETECT")
    try:
        d         = request.json
        img_array = b64_to_array(d["img"])
        face      = detect_and_crop_face(img_array)
        if face is None:
            return jsonify({"ok": False, "emotion": "neutral", "msg": "No face detected. Move closer."})
        tmp = save_temp(face)
        try:
            result = DeepFace.analyze(
                img_path=tmp, actions=["emotion"],
                enforce_detection=False, detector_backend="skip", silent=True
            )
            if isinstance(result, list):
                result = result[0]
            emotions_raw   = result.get("emotion", {})
            emotions_clean = {k: to_float(v) for k, v in emotions_raw.items()}
            emotion        = smart_emotion(emotions_raw)
            print(f"  🏆 Final emotion: {emotion}")
            return jsonify({
                "ok":      True,
                "emotion": emotion,
                "emotions": emotions_clean,
                "scores": {
                    "happy":    emotions_clean.get("happy",    0),
                    "sad":      emotions_clean.get("sad",      0),
                    "angry":    emotions_clean.get("angry",    0),
                    "fear":     emotions_clean.get("fear",     0),
                    "neutral":  emotions_clean.get("neutral",  0),
                    "surprise": emotions_clean.get("surprise", 0),
                }
            })
        except Exception as e:
            print(f"  ⚠️ Emotion error: {e}")
            return jsonify({"ok": False, "emotion": "neutral", "msg": str(e)})
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    except Exception as e:
        print(f"  ⚠️ Error: {e}")
        return jsonify({"ok": False, "emotion": "neutral", "msg": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 45)
    print(f"  Face Auth Server — port {port}")
    print(f"  DB path: {DB}")
    print("=" * 45)
    app.run(host="0.0.0.0", port=port, debug=False)