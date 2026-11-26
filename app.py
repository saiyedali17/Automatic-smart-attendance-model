"""Web version of the face-recognition attendance system.

The browser captures webcam frames (getUserMedia) and posts them here.
Detection: Haar cascade.  Recognition: LBPH (same algorithm as the desktop app).
"""
import base64
import csv
import datetime
import os
import re
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response, send_file

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(BASE, "data"))
TRAIN_DIR = os.path.join(DATA, "TrainingImage")
MODEL = os.path.join(DATA, "Trainner.yml")
STUDENTS = os.path.join(DATA, "studentdetails.csv")
ATT_DIR = os.path.join(DATA, "Attendance")
HAAR = os.path.join(BASE, "haarcascade_frontalface_default.xml")
THRESHOLD = float(os.environ.get("THRESHOLD", "65"))  # lower LBPH distance = better match
FACE_SIZE = (200, 200)
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")  # optional: protects the whole app

for d in (TRAIN_DIR, ATT_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
detector = cv2.CascadeClassifier(HAAR)
lock = threading.Lock()
_recognizer = None
_model_mtime = None


@app.before_request
def check_auth():
    if not APP_PASSWORD or request.path == "/health":
        return None
    auth = request.authorization
    if not auth or auth.password != APP_PASSWORD:
        return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Attendance"'})
    return None


def decode_image(data_url):
    raw = data_url.split(",", 1)[-1]
    arr = np.frombuffer(base64.b64decode(raw), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("bad image")
    return img


def detect_faces(gray):
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
    return gray, faces


def students():
    out = {}
    if os.path.exists(STUDENTS):
        with open(STUDENTS, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0].isdigit():
                    out[int(row[0])] = row[1]
    return out


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9 _-]", "", s).strip()[:40]


def get_recognizer():
    global _recognizer, _model_mtime
    if not os.path.exists(MODEL):
        return None
    m = os.path.getmtime(MODEL)
    if _recognizer is None or m != _model_mtime:
        r = cv2.face.LBPHFaceRecognizer_create()
        r.read(MODEL)
        _recognizer, _model_mtime = r, m
    return _recognizer


@app.get("/health")
def health():
    return "ok"


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/register/frame")
def register_frame():
    j = request.get_json(force=True)
    roll, name = str(j.get("roll", "")).strip(), safe_name(str(j.get("name", "")))
    if not roll.isdigit() or not name:
        return jsonify(ok=False, error="Roll number (digits) and name are required"), 400
    try:
        img = decode_image(j["image"])
    except Exception:
        return jsonify(ok=False, error="Bad image"), 400
    gray, faces = detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if len(faces) == 0:
        return jsonify(ok=True, saved=False, count=count_samples(roll))
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    existing = students()
    if int(roll) in existing and existing[int(roll)] != name:
        return jsonify(ok=False, error=f"Roll {roll} already belongs to another student"), 409
    folder = os.path.join(TRAIN_DIR, f"{roll}_{name}")
    os.makedirs(folder, exist_ok=True)
    n = len(os.listdir(folder)) + 1
    face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
    cv2.imwrite(os.path.join(folder, f"{name}_{roll}_{n}.jpg"), face)
    return jsonify(ok=True, saved=True, count=n, box=[int(x), int(y), int(w), int(h)])


def count_samples(roll):
    for d in os.listdir(TRAIN_DIR):
        if d.startswith(roll + "_"):
            return len(os.listdir(os.path.join(TRAIN_DIR, d)))
    return 0


@app.post("/api/register/finish")
def register_finish():
    j = request.get_json(force=True)
    roll, name = str(j.get("roll", "")).strip(), safe_name(str(j.get("name", "")))
    if not roll.isdigit() or not name or count_samples(roll) == 0:
        return jsonify(ok=False, error="No images captured for this student"), 400
    with lock:
        if int(roll) not in students():
            with open(STUDENTS, "a", newline="") as f:
                csv.writer(f).writerow([roll, name])
    return jsonify(ok=True)


@app.post("/api/train")
def train():
    faces, ids = [], []
    for d in sorted(os.listdir(TRAIN_DIR)):
        p = os.path.join(TRAIN_DIR, d)
        if not os.path.isdir(p) or not d.split("_")[0].isdigit():
            continue
        for fn in os.listdir(p):
            im = cv2.imread(os.path.join(p, fn), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                faces.append(cv2.resize(im, FACE_SIZE))
                ids.append(int(d.split("_")[0]))
    if not faces:
        return jsonify(ok=False, error="No training images. Register a student first."), 400
    with lock:
        r = cv2.face.LBPHFaceRecognizer_create()
        r.train(faces, np.array(ids))
        r.save(MODEL)
    return jsonify(ok=True, images=len(faces), students=len(set(ids)))


@app.post("/api/recognize")
def recognize():
    rec = get_recognizer()
    if rec is None:
        return jsonify(ok=False, error="Model not trained yet. Train it first."), 400
    try:
        img = decode_image(request.get_json(force=True)["image"])
    except Exception:
        return jsonify(ok=False, error="Bad image"), 400
    gray, faces = detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    names, out = students(), []
    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
        with lock:
            pid, conf = rec.predict(face)
        known = conf < THRESHOLD and pid in names
        out.append(dict(box=[int(x), int(y), int(w), int(h)], id=int(pid) if known else None,
                        name=names.get(pid) if known else "Unknown", conf=round(float(conf), 1)))
    return jsonify(ok=True, faces=out)


@app.post("/api/attendance/save")
def save_attendance():
    j = request.get_json(force=True)
    subject = safe_name(str(j.get("subject", "")))
    ids = sorted({int(i) for i in j.get("ids", []) if str(i).isdigit()})
    if not subject:
        return jsonify(ok=False, error="Subject name required"), 400
    names, now = students(), datetime.datetime.now()
    folder = os.path.join(ATT_DIR, subject)
    os.makedirs(folder, exist_ok=True)
    fn = f"{subject}_{now:%Y-%m-%d_%H-%M-%S}.csv"
    with open(os.path.join(folder, fn), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Enrollment", "Name", "Date", "Time"])
        for i in ids:
            w.writerow([i, names.get(i, ""), f"{now:%Y-%m-%d}", f"{now:%H:%M:%S}"])
    return jsonify(ok=True, file=fn, present=len(ids))


@app.get("/api/subjects")
def subjects():
    return jsonify(sorted(d for d in os.listdir(ATT_DIR) if os.path.isdir(os.path.join(ATT_DIR, d))))


@app.get("/api/attendance/<subject>")
def view_attendance(subject):
    folder = os.path.join(ATT_DIR, safe_name(subject))
    rows = []
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder), reverse=True):
            with open(os.path.join(folder, fn), newline="") as f:
                rows += [dict(r, Session=fn[:-4]) for r in csv.DictReader(f)]
    return jsonify(rows)


@app.get("/api/attendance/<subject>/download")
def download(subject):
    folder = os.path.join(ATT_DIR, safe_name(subject))
    out = os.path.join(DATA, f"{safe_name(subject)}_all.csv")
    with open(out, "w", newline="") as o:
        w = csv.writer(o)
        w.writerow(["Session", "Enrollment", "Name", "Date", "Time"])
        if os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                with open(os.path.join(folder, fn), newline="") as f:
                    for r in csv.DictReader(f):
                        w.writerow([fn[:-4], r["Enrollment"], r["Name"], r["Date"], r["Time"]])
    return send_file(out, as_attachment=True, download_name=f"{safe_name(subject)}_attendance.csv")


@app.get("/api/students")
def list_students():
    return jsonify([{"id": k, "name": v} for k, v in sorted(students().items())])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
