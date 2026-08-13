"""
Proctored exam app.

Flow: a host starts an exam session and shares the join link. Students open it,
sign in with enrollment number + password, pass a face check against their
registered photo, and enter the exam. From then on their browser uploads a frame
every few seconds; those frames both feed the cheating detector and render the
host's monitoring grid.

Video design: no WebRTC mesh. The host's grid is built from the frames students
already upload for detection, so there is nothing to negotiate and no relay to
configure. WebRTC is used only when the host opens one student for a live look.
"""
import os
import random
import threading
import time

import cv2
from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room as sio_join_room

from proctor.db_manager import db_manager
from proctor.face_engine import face_engine
from proctor.detect import cheating_detector

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(ROOT_DIR, "registered_faces")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "proctor-exam-dev-secret")
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*",
                    max_http_buffer_size=5 * 1024 * 1024)

# exams = { CODE: {"host_sid", "title", "started_at", "students": {sid: {...}}} }
exams = {}

CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no ambiguous characters

FRAME_INTERVAL_SECS = 3.0     # how often a student uploads a frame
STALL_AFTER_SECS = 20.0       # no frames for this long means detection is dead

# sid -> newest frame awaiting detection. A single worker drains this, so a slow
# machine drops stale frames instead of accumulating an unbounded backlog.
_pending = {}
_pending_lock = threading.Lock()


def generate_code():
    while True:
        code = "".join(random.choice(CODE_CHARS) for _ in range(6))
        if code not in exams:
            return code


def sid():
    return request.sid


def find_exam_by_sid(socket_id):
    """Returns (code, exam) for whichever exam this socket belongs to."""
    for code, exam in exams.items():
        if exam["host_sid"] == socket_id or socket_id in exam["students"]:
            return code, exam
    return None, None


# ---------------------------------------------------------------- pages

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "detectors": cheating_detector.status(),
        "active_exams": len(exams),
    })


# ---------------------------------------------------------------- registration

@app.route("/api/register", methods=["POST"])
def api_register():
    """Enrolls a student: credentials plus a reference face encoding."""
    data = request.get_json() or {}
    enrollment = (data.get("enrollment_number") or "").strip().upper()
    name = (data.get("name") or "").strip()
    password = data.get("password") or ""
    image = data.get("image_data") or ""

    if not enrollment or not name or not password:
        return jsonify({"ok": False, "error": "Enrollment number, name and password are required."}), 400
    if not image:
        return jsonify({"ok": False, "error": "A face photo is required."}), 400
    if db_manager.get_user_by_enrollment(enrollment):
        return jsonify({"ok": False, "error": "That enrollment number is already registered."}), 400

    ok, encoding, err = face_engine.extract_robust_encoding(image)
    if not ok:
        return jsonify({"ok": False, "error": err}), 400

    # Keep the reference photo alongside the encoding. Matching only needs the
    # encoding, but a human reviewing a disputed result needs to see the face
    # the system was comparing against.
    image_path = None
    try:
        os.makedirs(FACES_DIR, exist_ok=True)
        rel_path = os.path.join("registered_faces", f"{enrollment}.jpg")
        cv2.imwrite(os.path.join(ROOT_DIR, rel_path), face_engine.decode_base64_image(image))
        image_path = rel_path
    except Exception as e:
        # A missing photo must not block enrollment; the encoding is what matters.
        print(f"[Register] could not save reference photo for {enrollment}: {e}")

    db_manager.create_user_credentials(enrollment, password, name)
    db_manager.store_face_record(enrollment, encoding, image_path=image_path)
    return jsonify({"ok": True, "message": f"{name} registered."})


# ---------------------------------------------------------------- host

@socketio.on("host-start-exam")
def on_host_start_exam(data):
    data = data or {}
    title = (data.get("title") or "").strip() or "Examination"

    code = generate_code()
    exams[code] = {
        "host_sid": sid(),
        "title": title,
        "started_at": time.time(),
        "students": {},
    }
    sio_join_room(code)

    return {
        "ok": True,
        "code": code,
        "title": title,
        "join_url": f"{request.headers.get('Origin', '')}/?code={code}",
        "detectors": cheating_detector.status(),
    }


@socketio.on("host-end-exam")
def on_host_end_exam():
    code, exam = find_exam_by_sid(sid())
    if not exam or exam["host_sid"] != sid():
        return {"ok": False, "error": "Only the host can end the exam."}

    socketio.emit("exam-ended", {"reason": "The proctor ended this exam."}, room=code)
    for student_sid in list(exam["students"]):
        cheating_detector.remove_session(exam["students"][student_sid]["enrollment"])
        _pending.pop(student_sid, None)
    exams.pop(code, None)
    return {"ok": True}


@socketio.on("host-remove-student")
def on_host_remove_student(data):
    code, exam = find_exam_by_sid(sid())
    if not exam or exam["host_sid"] != sid():
        return {"ok": False, "error": "Only the host can remove a student."}

    target = (data or {}).get("student_id")
    student = exam["students"].get(target)
    if not student:
        return {"ok": False, "error": "That student is no longer in the exam."}

    remove_student(code, target, "You were removed from the exam by the proctor.")
    return {"ok": True}


# ---------------------------------------------------------------- student

@socketio.on("student-join")
def on_student_join(data):
    """Credentials, then face match, then admission - in that order."""
    data = data or {}
    code = (data.get("code") or "").strip().upper()
    enrollment = (data.get("enrollment_number") or "").strip().upper()
    password = data.get("password") or ""
    image = data.get("image_data") or ""

    exam = exams.get(code)
    if not exam:
        return {"ok": False, "error": "No exam found with that code."}

    user = db_manager.verify_user_credentials(enrollment, password)
    if not user:
        return {"ok": False, "error": "Incorrect enrollment number or password."}

    for existing in exam["students"].values():
        if existing["enrollment"] == enrollment:
            return {"ok": False, "error": "You are already signed in to this exam."}

    face_record = db_manager.get_face_record(enrollment)
    if not face_record:
        return {"ok": False, "error": "No registered face found. Please register first."}

    if not image:
        return {"ok": False, "error": "A camera snapshot is required."}

    matched, distance, message = face_engine.verify_face(image, face_record["encoding"])
    if not matched:
        return {"ok": False, "error": f"Face verification failed: {message}"}

    exam["students"][sid()] = {
        "enrollment": enrollment,
        "name": user["name"],
        "joined_at": time.time(),
        "warnings": 0,
        "last_frame_at": None,
        "last_frame": None,
        "last_flags": None,
        "last_hands": [],
    }
    sio_join_room(code)
    cheating_detector.register_session(enrollment)

    socketio.emit("student-joined", {
        "student_id": sid(),
        "name": user["name"],
        "enrollment": enrollment,
    }, room=exam["host_sid"])

    return {
        "ok": True,
        "code": code,
        "title": exam["title"],
        "name": user["name"],
        "enrollment": enrollment,
        "frame_interval": FRAME_INTERVAL_SECS,
    }


@socketio.on("student-frame")
def on_student_frame(data):
    """One webcam frame: feeds both the host's grid and the detector."""
    code, exam = find_exam_by_sid(sid())
    if not exam or sid() not in exam["students"]:
        return

    image = (data or {}).get("image_data")
    if not image:
        return

    student = exam["students"][sid()]
    student["last_frame_at"] = time.time()
    student["last_frame"] = image
    # Verdicts and hand boxes from the proctor-x engine in the candidate's browser.
    # Kept as None when the browser engine is not running, so the detector knows
    # to fall back rather than reading "no flags" as "nothing is wrong".
    student["last_flags"] = (data or {}).get("flags")
    student["last_hands"] = (data or {}).get("hands") or []

    # The host's monitoring grid is just these frames - no WebRTC needed.
    socketio.emit("student-frame", {
        "student_id": sid(),
        "name": student["name"],
        "enrollment": student["enrollment"],
        "image_data": image,
    }, room=exam["host_sid"])

    # Detection runs on a worker; only the newest frame per student is kept.
    with _pending_lock:
        _pending[sid()] = {"code": code, "enrollment": student["enrollment"]}


# ---------------------------------------------------------------- detection worker

def analyse_frame(student_sid, job):
    exam = exams.get(job["code"])
    if not exam:
        return
    student = exam["students"].get(student_sid)
    if not student or not student["last_frame"]:
        return

    result = cheating_detector.analyze(
        job["enrollment"],
        student["last_frame"],
        client_flags=student.get("last_flags"),
        hand_boxes=student.get("last_hands"),
    )
    if not result.get("ok"):
        return

    # A background thread has no request context, so every send needs an
    # explicit target via socketio.emit rather than flask_socketio.emit.
    socketio.emit("proctor-status", {
        "cheating": result["cheating"],
        "warning_count": result["warning_count"],
        "reasons": result.get("reasons", []),
    }, room=student_sid)

    if not result.get("warning_issued"):
        return

    student["warnings"] = result["warning_count"]

    socketio.emit("proctor-warning", {
        "warning_count": result["warning_count"],
        "reasons": result["reasons"],
        "removed": result["kick"],
    }, room=student_sid)

    socketio.emit("proctor-alert", {
        "student_id": student_sid,
        "name": student["name"],
        "enrollment": student["enrollment"],
        "warning_count": result["warning_count"],
        "reasons": result["reasons"],
        "removed": result["kick"],
        "at": time.time(),
    }, room=exam["host_sid"])

    if result["kick"]:
        remove_student(job["code"], student_sid,
                       "Removed from the exam after 3 cheating warnings.")


def detection_worker():
    while True:
        socketio.sleep(0.25)
        with _pending_lock:
            jobs = list(_pending.items())
            _pending.clear()
        for student_sid, job in jobs:
            try:
                analyse_frame(student_sid, job)
            except Exception as e:
                print(f"[Detection] failed for {student_sid}: {e}")


def stall_watchdog():
    """
    Reports students whose frames stopped arriving.

    The client skips a frame whenever its video element has no dimensions, so
    detection can quietly stop while everything still looks fine. Missing frames
    are themselves worth reporting.
    """
    reported = set()
    while True:
        socketio.sleep(10)
        now = time.time()
        for code, exam in list(exams.items()):
            for student_sid, student in list(exam["students"].items()):
                last = student.get("last_frame_at")
                if last is None or now - last <= STALL_AFTER_SECS:
                    reported.discard(student_sid)
                    continue
                if student_sid in reported:
                    continue
                reported.add(student_sid)
                socketio.emit("proctor-stalled", {
                    "student_id": student_sid,
                    "name": student["name"],
                    "stalled_seconds": int(now - last),
                }, room=exam["host_sid"])


# ---------------------------------------------------------------- live view (WebRTC)

@socketio.on("live-view-request")
def on_live_view_request(data):
    """Host asks one student to open a live stream."""
    code, exam = find_exam_by_sid(sid())
    if not exam or exam["host_sid"] != sid():
        return {"ok": False, "error": "Only the host can open a live view."}

    target = (data or {}).get("student_id")
    if target not in exam["students"]:
        return {"ok": False, "error": "That student is no longer in the exam."}

    socketio.emit("live-view-request", {"host_id": sid()}, room=target)
    return {"ok": True}


@socketio.on("live-view-stop")
def on_live_view_stop(data):
    code, exam = find_exam_by_sid(sid())
    if not exam or exam["host_sid"] != sid():
        return
    target = (data or {}).get("student_id")
    if target:
        socketio.emit("live-view-stop", {}, room=target)


@socketio.on("signal")
def on_signal(data):
    """Relays WebRTC offer/answer/ICE between the host and one student."""
    to = (data or {}).get("to")
    payload = (data or {}).get("data")
    if to and payload:
        socketio.emit("signal", {"from": sid(), "data": payload}, room=to)


@app.route("/api/ice-config")
def api_ice_config():
    """
    ICE servers for the live view, with TURN supplied by environment.

    STUN only discovers an address; it cannot relay. Peers behind symmetric NAT
    or CGNAT (campus wifi, mobile hotspots) need a TURN relay or the stream never
    carries media. Set TURN_URL / TURN_USER / TURN_PASS to enable it. The
    monitoring grid does not depend on any of this.
    """
    ice = [{"urls": "stun:stun.l.google.com:19302"},
           {"urls": "stun:stun1.l.google.com:19302"}]

    url = os.environ.get("TURN_URL", "").strip()
    user = os.environ.get("TURN_USER", "").strip()
    password = os.environ.get("TURN_PASS", "").strip()

    if url and user and password:
        ice.append({"urls": [u.strip() for u in url.split(",") if u.strip()],
                    "username": user, "credential": password})
        return jsonify({"iceServers": ice, "relayReady": True})

    return jsonify({"iceServers": ice, "relayReady": False})


# ---------------------------------------------------------------- teardown

def remove_student(code, student_sid, reason):
    exam = exams.get(code)
    if not exam:
        return
    student = exam["students"].pop(student_sid, None)
    if not student:
        return

    cheating_detector.remove_session(student["enrollment"])
    _pending.pop(student_sid, None)

    socketio.emit("exam-ended", {"reason": reason}, room=student_sid)
    socketio.emit("student-left", {
        "student_id": student_sid,
        "name": student["name"],
        "reason": reason,
    }, room=exam["host_sid"])


@socketio.on("disconnect")
def on_disconnect():
    code, exam = find_exam_by_sid(sid())
    if not exam:
        return

    if exam["host_sid"] == sid():
        socketio.emit("exam-ended", {"reason": "The proctor disconnected."}, room=code)
        for student_sid, student in list(exam["students"].items()):
            cheating_detector.remove_session(student["enrollment"])
            _pending.pop(student_sid, None)
        exams.pop(code, None)
    else:
        remove_student(code, sid(), "You left the exam.")


db_manager.init_databases()
socketio.start_background_task(detection_worker)
socketio.start_background_task(stall_watchdog)

if __name__ == "__main__":
    # Not 5000: macOS AirPlay Receiver squats on it and answers with a 403.
    port = int(os.environ.get("PORT", 5050))
    print(f"Proctored exam app running on http://0.0.0.0:{port}")
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
