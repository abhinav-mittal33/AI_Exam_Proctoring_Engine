# Proctor — online examination with live proctoring

A proctor starts an exam and shares a link. Candidates open it, sign in with their
enrollment number and password, pass a face match against their registered photo,
and enter the exam. From then on their browser uploads a frame every few seconds;
those frames both feed the cheating detector and paint the proctor's monitoring grid.

## Why there is no video mesh

The monitoring grid is built from the frames candidates already upload for
detection. Nothing is negotiated, so there is no TURN server to configure, no NAT
traversal to fail, and no black tiles — and it stays cheap with 30+ candidates.

WebRTC is used in exactly one place: when the proctor clicks a candidate to open a
live stream. That path benefits from a TURN relay (see below), and if it fails the
grid keeps working regardless.

## Running it

```bash
pip install -r requirements.txt
```
```bash
python app.py
```

Open http://localhost:5050.

1. **Register a face** — "Register your face" on the landing page. Name, enrollment
   number, password, and a webcam photo. This is the reference for sign-in.
2. **Start an exam** — click "Start an exam" as the proctor and copy the join link.
3. **Join** — candidates open the link, sign in, and pass the face check.
4. **Monitor** — candidates appear in the grid, alerts stream into the side panel,
   and clicking a tile opens a live view.

## What counts as cheating

Detection is the proctor-x engine, split across the browser and the server.

**In the candidate's browser** (`static/js/proctorx.js`) — MediaPipe FaceLandmarker
(478 landmarks, iris, and a 3D facial transformation matrix) plus HandLandmarker:

| Signal | How |
|---|---|
| Head turned / looking down | True 3D pose from the transformation matrix, EMA-smoothed |
| Eye gaze, with direction | Iris position within each eye, measured from the candidate's resting gaze; reported as looking left / right / up / down |
| Talking | Mouth aspect ratio plus its variance over a short window |
| Eyes closed | Eye aspect ratio |
| Hands | Hand boxes, so the server can tell a held phone from one on a shelf |

This runs client-side deliberately: landmark extraction is the expensive part, and
keeping it off the server is what lets one small instance host many candidates.

**On the server** (`proctor/detect.py`) — the checks that must not be forgeable,
since anything computed in a candidate's own browser can be tampered with:

| Signal | How |
|---|---|
| Phone, book, laptop | YOLOv8n via onnxruntime, marked "held" when it overlaps a hand |
| Face count (absent / multiple) | YuNet |

Browser verdicts arrive already debounced by hysteresis buffers: a signal must hold
for several consecutive frames to count, and decays faster than it builds, so one
glance away is not a warning.

### Calibration

The first ~3 seconds of an exam learn the candidate's resting posture (median yaw,
pitch, roll, mouth opening and eye opening). Every pose threshold afterwards is a
deviation from *their* neutral, not an absolute angle. Absolute limits punished
anyone who naturally sits at an angle or has a webcam mounted off-centre, which was
the main source of false positives.

| Signal | Fires at | Sustained for |
|---|---|---|
| Head turned | 22° off resting yaw | 12 frames (~0.8s) |
| Looking down | 18° off resting pitch | 14 frames (~0.9s) |
| Head tilted sideways | 20° off resting roll | 14 frames (~0.9s) |
| Eyes off screen (with direction) | 0.16 iris offset from resting gaze | 14 frames |
| Talking | 3 mouth open/close cycles in 2.5s | 4 frames |
| Second person | any extra face | 3 frames (~0.2s) |
| Eyes closed | 62% of resting opening | 18 frames |

Talking is counted as *cycles* rather than raw mouth openness. Quiet speech barely
opens the mouth and was being missed entirely by a fixed openness threshold, while a
single yawn — one long opening — no longer counts, because it is one cycle, not several.

### Severity

Violations carry a severity that decides how soon the same one may be reported again:

| Severity | Examples | Repeat after |
|---|---|---|
| CRITICAL | second person, identity mismatch | 6s |
| HIGH | phone or book, talking, tab switch | 12s |
| MEDIUM | head turned, looking down, head tilt | 25s |
| LOW | eyes closed | 60s |

Three warnings removes a candidate automatically.

If the browser engine fails to load, the server falls back to estimating pose from
YuNet's five points. That fallback is noticeably noisier — it is a safety net, not
the intended path.

## Layout

```
app.py                 Flask + Socket.IO: exam rooms, auth, frames, alerts
proctor/
  face_engine.py       YuNet detection + SFace recognition (identity)
  detect.py            Cheating detection engine
  db_manager.py        SQLite: credentials and face encodings
models/                ONNX weights (YuNet, SFace, YOLOv8n)
static/                Frontend (single page, vanilla JS)
  js/proctorx.js       Proctor-X engine: MediaPipe landmarks, gaze, hysteresis
data/                  SQLite databases - gitignored, never commit
registered_faces/      Reference photos - gitignored, never commit
```

## Configuration

| Variable | Purpose |
|---|---|
| `PORT` | HTTP port (default 5050) |
| `SECRET_KEY` | Flask session secret |
| `TURN_URL` / `TURN_USER` / `TURN_PASS` | TURN relay for the live view only |

Without TURN, the live view may fail between different networks; the grid is
unaffected. Check `/api/ice-config` — `relayReady` tells you which state you are in.

## Health

`GET /health` reports detector status:

```json
{"status": "ok", "detectors": {"object_detection": true, "face_detection": true}}
```

If `object_detection` is ever `false`, phone and book detection is **not running**.
The proctor's screen shows this as a red "Object detection OFF" pill rather than
failing quietly.

## Notes

- `data/` holds password hashes and biometric face encodings. It is gitignored and
  must stay that way.
- Candidates see only their own camera. Only the proctor sees the full grid.
- If a candidate's frames stop arriving (camera off or blocked), the proctor is told
  after 20s rather than the tile simply going stale.
