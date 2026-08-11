# AI Exam Proctoring Engine

A modular, evidence-based **AI Exam Proctoring Engine** built using **FastAPI**, **MediaPipe**, **OpenCV**, **PyYAML**, **WebSockets**, and **React**.

The system continuously monitors student video/audio streams during online examinations, detects observable computer vision signals, debounces events over rolling temporal windows, computes dynamic evidence risk scores, and streams live events to a Reviewer Dashboard.

---

## 🏗️ Architectural Separation

The Proctoring Engine operates independently from student identity authentication:

```text
┌─────────────────────────────────────┐
│      STUDENT IDENTITY ENGINE        │
│    (Face_Recognition_Arcface)       │
│                                     │
│ Enrollment                           │
│ Password / Argon2id                  │
│ ArcFace 512-dim Biometrics           │
│ Liveness Anti-Spoofing               │
└──────────────────┬──────────────────┘
                   │
                   │ VERIFIED SESSION (session_id)
                   ▼
┌─────────────────────────────────────┐
│        PROCTORING ENGINE            │
│    (exam-cheating-detection)        │
│                                     │
│ Multi-Rate Vision Pipeline          │
│ Face Presence (30 FPS)              │
│ Head Pose & Gaze (10 FPS)           │
│ Mouth Movement (10 FPS)             │
│ Object Detection (5 FPS)            │
│ Audio VAD & Energy                  │
│ Temporal Event Debouncer            │
│ Evidence Risk Scoring               │
└──────────────────┬──────────────────┘
                   │
                   ▼
       OBSERVABLE PROCTOR EVENTS
                   │
                   ▼
       REVIEWER DASHBOARD (port 3001)
```

---

## ⚡ Multi-Rate Processing Pipeline

- **30 FPS**: Face presence detection, face count tracking (0, 1, 2+), bounding box tracking, grace period handling (`FACE_MISSING_GRACE_PERIOD = 3.0s`).
- **10 FPS**: 3D PnP Head pose estimation (`yaw`, `pitch`, `roll`), Gaze direction displacement (`GAZE_AWAY`), Mouth Aspect Ratio (`MAR`).
- **5 FPS**: Prohibited object detection (`cell phone`, `book`, `laptop`, `remote`).
- **Audio VAD**: Spectral & RMS energy Voice Activity Detection (`AUDIO_ACTIVITY`).

---

## 🛡️ Observable Proctoring Events

The engine does NOT output simplistic labels like `"CHEATING"`. It produces evidence-based, observable events:

- `FACE_MISSING`: No face detected in camera frame beyond grace period.
- `MULTIPLE_FACES`: More than one person detected in frame.
- `HEAD_TURNED_LEFT` / `HEAD_TURNED_RIGHT` / `HEAD_TURNED_UP` / `HEAD_TURNED_DOWN`: Significant head rotation angle.
- `GAZE_AWAY`: Sustained pupil/iris gaze deviation away from screen center.
- `MOUTH_MOVEMENT` / `POSSIBLE_SPEECH`: Sustained mouth aspect ratio change.
- `PHONE_DETECTED`: Cell phone object detected in video frame.
- `PROHIBITED_OBJECT_DETECTED`: Prohibited exam item detected.
- `AUDIO_ACTIVITY`: Voice activity detected by audio monitor.

---

## ⚖️ Evidence Risk Engine

- Dynamic score calculation (0.0 to 100.0) based on weighted event severities and durations.
- Smooth temporal score decay during periods of normal behavior.
- Categorized Risk Levels:
  - `NORMAL` (0 - 20 pts)
  - `LOW_RISK` (21 - 40 pts)
  - `REVIEW` (41 - 65 pts)
  - `HIGH_PRIORITY_REVIEW` (66+ pts)

---

## 🚀 Quickstart & Setup

### 1. Backend Server (FastAPI + WebSockets)
```bash
cd "/Users/abhinavmittal/Temperory/face /exam-cheating-detection"
/Users/abhinavmittal/Temperory/face\ /Face_Recognition_Arcface/venv/bin/uvicorn backend.main:app --port 8001
```
FastAPI Swagger OpenAPI Docs: **[http://localhost:8001/docs](http://localhost:8001/docs)**
Health Check: **[http://localhost:8001/health](http://localhost:8001/health)**

### 2. Reviewer Dashboard Frontend (React)
```bash
cd "/Users/abhinavmittal/Temperory/face /exam-cheating-detection/frontend"
npm start
```
Reviewer Dashboard: **[http://localhost:3001](http://localhost:3001)**

### 3. Run Automated Tests
```bash
cd "/Users/abhinavmittal/Temperory/face /exam-cheating-detection"
PYTHONPATH=. /Users/abhinavmittal/Temperory/face\ /Face_Recognition_Arcface/venv/bin/python tests/test_proctoring.py
```
