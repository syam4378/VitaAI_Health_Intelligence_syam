"""
VitaAI Face Server — Final Production Grade
  ✅ SCRFD ONNX detection (det_10g from buffalo_l)
  ✅ ArcFace w600k_r50 512-D embeddings
  ✅ FERPlus-8 ONNX emotion (stable ONNX model zoo — 8 classes)
  ✅ NMS, strict alignment, blur/size/multi-face validation
  ✅ MongoDB with timestamp + model version
  ✅ L2 normalization + cosine similarity
"""

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
CORS(app, origins=["*"], supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD"
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        from flask import make_response
        res = make_response("", 200)
        res.headers["Access-Control-Allow-Origin"]  = "*"
        res.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD"
        return res

# ── CONFIG ────────────────────────────────────────────────
MODELS_DIR      = "/app/models"
SCRFD_PATH      = f"{MODELS_DIR}/scrfd_2.5g_bnkps.onnx"
ARCFACE_PATH    = f"{MODELS_DIR}/arcface_w600k_r50.onnx"
EMOTION_PATH    = f"{MODELS_DIR}/emotion_miniXception.onnx"
MODEL_VERSION   = "scrfd+arcface+v1"

SIMILARITY_THRESHOLD = 0.45
MIN_FACE_SIZE        = 40
BLUR_THRESHOLD       = 50.0

# FERPlus-8 label order (neutral=0, happiness=1, surprise=2, sadness=3,
#                         anger=4, disgust=5, fear=6, contempt=7)
EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]

# ArcFace 5-point reference landmarks for 112×112
ARCFACE_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# ── MONGODB ───────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "")
_client   = None
_col      = None

def get_col():
    global _client, _col
    if _col is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _col    = _client["vitaai"]["faces"]
        _col.create_index("email", unique=True)
    return _col

def db_load(email: str):
    doc = get_col().find_one({"email": email}, {"embeddings": 1})
    if doc and "embeddings" in doc:
        return [np.array(e["vec"], dtype=np.float32) for e in doc["embeddings"]]
    return None

def db_save(email: str, embedding: np.ndarray):
    doc = get_col().find_one({"email": email}, {"embeddings": 1})
    emb_entry = {
        "vec":   embedding.tolist(),
        "ts":    datetime.utcnow().isoformat(),
        "model": MODEL_VERSION,
    }
    if doc and "embeddings" in doc:
        existing = doc["embeddings"]
        if len(existing) >= 3:
            existing = existing[-2:]
        existing.append(emb_entry)
        get_col().update_one({"email": email}, {"$set": {"embeddings": existing}})
    else:
        get_col().update_one(
            {"email": email},
            {"$set": {"email": email, "embeddings": [emb_entry]}},
            upsert=True,
        )

def db_count() -> int:
    try:
        return get_col().count_documents({})
    except Exception:
        return -1

# ── LOAD MODELS ───────────────────────────────────────────
scrfd_sess   = None
arcface_sess = None
emotion_sess = None

def load_models():
    global scrfd_sess, arcface_sess, emotion_sess
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2

    if os.path.exists(SCRFD_PATH):
        scrfd_sess = ort.InferenceSession(SCRFD_PATH, opts, providers=["CPUExecutionProvider"])
        print("✅ SCRFD ONNX loaded")
    else:
        print("⚠️  SCRFD missing — OpenCV Haar fallback active")

    if os.path.exists(ARCFACE_PATH):
        arcface_sess = ort.InferenceSession(ARCFACE_PATH, opts, providers=["CPUExecutionProvider"])
        print("✅ ArcFace ONNX loaded")
    else:
        print("❌ ArcFace model missing!")

    if os.path.exists(EMOTION_PATH):
        emotion_sess = ort.InferenceSession(EMOTION_PATH, opts, providers=["CPUExecutionProvider"])
        print("✅ Emotion FERPlus-8 ONNX loaded")
    else:
        print("⚠️  Emotion ONNX missing — variance fallback active")

load_models()

haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── IMAGE HELPERS ─────────────────────────────────────────
def b64_to_bgr(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",")[1]
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def check_blur(img_bgr: np.ndarray) -> bool:
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"  📊 Blur variance: {variance:.1f} (threshold={BLUR_THRESHOLD})")
    return variance >= BLUR_THRESHOLD

# ── NMS ───────────────────────────────────────────────────
def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1   = np.maximum(x1[i], x1[order[1:]])
        yy1   = np.maximum(y1[i], y1[order[1:]])
        xx2   = np.minimum(x2[i], x2[order[1:]])
        yy2   = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou   = inter / (union + 1e-10)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep

# ── SCRFD DETECTION ───────────────────────────────────────
def detect_faces(img_bgr: np.ndarray):
    if scrfd_sess is None:
        return _haar_detect(img_bgr)

    h, w   = img_bgr.shape[:2]
    scale  = min(640 / w, 640 / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (nw, nh))
    padded  = np.zeros((640, 640, 3), dtype=np.uint8)
    padded[:nh, :nw] = resized

    inp = padded.astype(np.float32)
    inp = (inp - 127.5) / 128.0
    inp = inp.transpose(2, 0, 1)[np.newaxis]

    input_name = scrfd_sess.get_inputs()[0].name
    try:
        outputs = scrfd_sess.run(None, {input_name: inp})
    except Exception as e:
        print(f"  ⚠️  SCRFD inference error: {e}")
        return _haar_detect(img_bgr)

    try:
        scores_raw = outputs[0].reshape(-1)
        boxes_raw  = outputs[1].reshape(-1, 4)
        lms_raw    = outputs[2].reshape(-1, 5, 2) if len(outputs) > 2 else None
    except Exception as e:
        print(f"  ⚠️  SCRFD output parse error: {e}")
        return _haar_detect(img_bgr)

    mask = scores_raw > 0.5
    if not np.any(mask):
        return _haar_detect(img_bgr)

    scores_f = scores_raw[mask]
    boxes_f  = boxes_raw[mask]
    lms_f    = lms_raw[mask] if lms_raw is not None else None

    keep      = nms(boxes_f, scores_f, iou_threshold=0.4)
    inv_scale = 1.0 / scale
    results   = []

    for i in keep:
        x1, y1, x2, y2 = boxes_f[i] * inv_scale
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        lms = lms_f[i] * inv_scale if lms_f is not None else None
        results.append({"box": (x1, y1, x2, y2), "score": float(scores_f[i]), "landmarks": lms})
        print(f"  ✅ Face: ({x1},{y1},{x2},{y2}) score={scores_f[i]:.3f}")

    return results

def _haar_detect(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for scale, neighbors in [(1.1, 5), (1.05, 3)]:
        faces = haar_cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(30, 30)
        )
        if len(faces) > 0:
            return [{"box": (x, y, x + fw, y + fh), "score": 0.8, "landmarks": None}
                    for x, y, fw, fh in faces]
    return []

# ── FACE ALIGNMENT ────────────────────────────────────────
def align_face(img_bgr: np.ndarray, landmarks: np.ndarray):
    src  = landmarks.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_REF, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img_bgr, M, (112, 112),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

# ── ARCFACE EMBEDDING ─────────────────────────────────────
def get_embedding(face_112: np.ndarray) -> np.ndarray:
    face_rgb  = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB).astype(np.float32)
    face_norm = (face_rgb - 127.5) / 128.0
    face_inp  = face_norm.transpose(2, 0, 1)[np.newaxis]
    inp_name  = arcface_sess.get_inputs()[0].name
    emb       = arcface_sess.run(None, {inp_name: face_inp})[0][0]
    norm      = np.linalg.norm(emb)
    emb       = emb / (norm + 1e-10)
    print(f"  ✅ Embedding shape={emb.shape}, L2={norm:.4f}")
    return emb.astype(np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def best_sim(stored: list, live: np.ndarray) -> float:
    sims = [cosine_sim(s, live) for s in stored]
    best = max(sims)
    print(f"  📊 Similarities: {[round(s, 4) for s in sims]} | Best: {best:.4f}")
    return best

# ── FULL FACE PIPELINE ────────────────────────────────────
def face_pipeline(img_bgr: np.ndarray, strict: bool = True):
    if arcface_sess is None:
        return None, "ArcFace model not loaded."

    if not check_blur(img_bgr):
        return None, "Image is too blurry. Ensure good lighting and hold steady."

    faces = detect_faces(img_bgr)
    if not faces:
        return None, "No face detected. Come closer and look straight at the camera."
    if len(faces) > 1:
        return None, "Multiple faces detected. Please ensure only one face is visible."

    face            = faces[0]
    x1, y1, x2, y2 = face["box"]
    fw, fh          = x2 - x1, y2 - y1

    if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
        return None, f"Face too small ({fw}×{fh} px). Please come closer to the camera."

    lms = face["landmarks"]
    if lms is not None:
        aligned = align_face(img_bgr, lms)
        if aligned is None:
            if strict:
                return None, "Face alignment failed. Look straight at the camera."
            aligned = cv2.resize(img_bgr[y1:y2, x1:x2], (112, 112))
        else:
            print("  ✅ Aligned via 5-point landmarks")
    else:
        if strict:
            return None, "Landmarks not detected. Ensure frontal, well-lit face."
        aligned = cv2.resize(img_bgr[y1:y2, x1:x2], (112, 112))

    emb = get_embedding(aligned)
    return emb, None

# ── EMOTION DETECTION ─────────────────────────────────────
def detect_emotion(img_bgr: np.ndarray) -> tuple:
    faces = detect_faces(img_bgr)
    if not faces:
        return "neutral", {}

    face            = max(faces, key=lambda f: f["score"])
    x1, y1, x2, y2 = face["box"]
    gray            = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    roi             = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return "neutral", {}

    if emotion_sess is not None:
        # FERPlus-8: input (1,1,64,64), float32, range 0-255 (no /255 normalization)
        face_64  = cv2.resize(roi, (64, 64)).astype(np.float32)
        face_inp = face_64[np.newaxis, np.newaxis]   # (1,1,64,64)
        inp_name = emotion_sess.get_inputs()[0].name
        preds    = emotion_sess.run(None, {inp_name: face_inp})[0][0]
        # Softmax
        preds    = np.exp(preds - preds.max())
        preds   /= preds.sum()
        scores   = {EMOTION_LABELS[i]: float(preds[i]) for i in range(len(EMOTION_LABELS))}
    else:
        var    = float(np.var(roi))
        bright = float(np.mean(roi))
        scores = {l: 0.0 for l in EMOTION_LABELS}
        if var > 1500 and bright > 100:
            scores["happy"]   = 1.0
        elif bright < 70:
            scores["sad"]     = 1.0
        else:
            scores["neutral"] = 1.0

    dominant = max(scores, key=scores.get)
    score    = scores[dominant]
    print(f"  😊 {dominant} ({score:.3f})")

    if dominant == "neutral" and score < 0.45:
        others = {k: v for k, v in scores.items() if k != "neutral"}
        if others:
            second = max(others, key=others.get)
            if scores[second] >= 0.20:
                print(f"  → Override neutral → {second}")
                return second, scores

    return dominant, scores

# ── ROUTES ────────────────────────────────────────────────

@app.route("/", methods=["GET", "HEAD", "OPTIONS"])
def index():
    return jsonify({
        "status":       "VitaAI Face Server ✅",
        "detection":    "SCRFD ONNX (det_10g)" if scrfd_sess   else "OpenCV Haar fallback",
        "recognition":  "ArcFace ONNX 512-D"   if arcface_sess else "❌ NOT LOADED",
        "emotion":      "FERPlus-8 ONNX"        if emotion_sess else "variance fallback",
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

    emb, err = face_pipeline(img_bgr, strict=True)
    if err:
        return jsonify({"ok": False, "msg": err})

    try:
        db_save(email, emb)
        doc   = get_col().find_one({"email": email}, {"embeddings": 1})
        count = len(doc["embeddings"]) if doc else 1
        msg   = "Face registered!" if count == 1 else f"Face updated! ({count}/3 angles stored)"
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

    live_emb, err = face_pipeline(img_bgr, strict=True)
    if err:
        return jsonify({"ok": False, "msg": err})

    sim = best_sim(stored, live_emb)
    ok  = sim >= SIMILARITY_THRESHOLD
    print(f"  {'✅ PASS' if ok else '❌ FAIL'} sim={sim:.4f}")

    return jsonify({
        "ok":  bool(ok),
        "msg": "Face verified! Welcome back 👋" if ok else "Face did not match. Please try again.",
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
    print(f"  Detection   : {'SCRFD ONNX' if scrfd_sess   else 'OpenCV Haar'}")
    print(f"  Recognition : {'ArcFace ONNX 512-D' if arcface_sess else '❌ NOT LOADED'}")
    print(f"  Emotion     : {'FERPlus-8 ONNX' if emotion_sess else 'fallback'}")
    print(f"  Threshold   : {SIMILARITY_THRESHOLD}")
    print(f"  Faces stored: {db_count()}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False)