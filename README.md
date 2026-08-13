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

Open http://localhost:5000.

1. **Register a face** — "Register your face" on the landing page. Name, enrollment
   number, password, and a webcam photo. This is the reference for sign-in.
2. **Start an exam** — click "Start an exam" as the proctor and copy the join link.
3. **Join** — candidates open the link, sign in, and pass the face check.
4. **Monitor** — candidates appear in the grid, alerts stream into the side panel,
   and clicking a tile opens a live view.

## What counts as cheating

Detection runs server-side on `onnxruntime` (deliberately not `ultralytics`, which
pulls in torch and does not fit a 512MB instance):

| Signal | Source |
|---|---|
| Phone, book, laptop in frame | YOLOv8n via onnxruntime |
| No face / multiple faces | YuNet face detector |
| Head turned or tilted down | Facial landmark geometry |
| Talking | Mouth aspect ratio over time |

Repeat detections of the same kind are suppressed for 12s, and pose signals need
several consecutive frames before they count, so one glance away is not a warning.
Three warnings removes a candidate from the exam automatically.

## Layout

```
app.py                 Flask + Socket.IO: exam rooms, auth, frames, alerts
proctor/
  face_engine.py       YuNet detection + SFace recognition (identity)
  detect.py            Cheating detection engine
  db_manager.py        SQLite: credentials and face encodings
models/                ONNX weights (YuNet, SFace, YOLOv8n)
static/                Frontend (single page, vanilla JS)
data/                  SQLite databases - gitignored, never commit
```

## Configuration

| Variable | Purpose |
|---|---|
| `PORT` | HTTP port (default 5000) |
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
