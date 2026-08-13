/*
 * Proctor-X detection engine, running in the candidate's browser.
 *
 * Ported from the proctor-x project: MediaPipe FaceLandmarker (478 landmarks,
 * iris, and a facial transformation matrix) plus HandLandmarker. This runs
 * client-side on purpose - landmark extraction is the expensive part, and doing
 * it here keeps the server able to host many candidates at once.
 *
 * What it produces each frame is a set of debounced flags. The server treats
 * those as hints only: face count and objects are re-checked server-side, since
 * anything computed in a candidate's own browser can be tampered with.
 */

const MEDIAPIPE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';
const MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models';

// ---------- geometry (ported from proctor-x lib/math.ts) ----------

function distance(p1, p2) {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}

/** Eye aspect ratio — low values mean the eyes are closing. */
function getEAR(lm) {
  const l1 = distance(lm[160], lm[144]);
  const l2 = distance(lm[158], lm[153]);
  const l3 = distance(lm[33], lm[133]);
  const leftEAR = (l1 + l2) / (2.0 * l3);

  const r1 = distance(lm[385], lm[380]);
  const r2 = distance(lm[387], lm[373]);
  const r3 = distance(lm[362], lm[263]);
  const rightEAR = (r1 + r2) / (2.0 * r3);

  return (leftEAR + rightEAR) / 2.0;
}

/** Mouth aspect ratio — opening relative to mouth width. */
function getMAR(lm) {
  const vertical = distance(lm[13], lm[14]);
  const horizontal = distance(lm[61], lm[291]);
  return horizontal === 0 ? 0 : vertical / horizontal;
}

/** True 3D head pose from MediaPipe's 4x4 column-major transformation matrix. */
function getHeadPose(matrix) {
  const R11 = matrix[0];
  const R21 = matrix[1];
  const R31 = matrix[2];
  const R32 = matrix[6];
  const R33 = matrix[10];

  const pitch = Math.atan2(-R31, Math.sqrt(R32 * R32 + R33 * R33)) * (180 / Math.PI);
  const yaw = Math.atan2(R21, R11) * (180 / Math.PI);
  const roll = Math.atan2(R32, R33) * (180 / Math.PI);

  return { pitch, yaw, roll };
}

/** Iris offset within each eye — where the eyes point, not where the head does. */
function getGaze(lm) {
  const leftIris = lm[468];
  const rightIris = lm[473];
  if (!leftIris || !rightIris) return { gazeYaw: 0, gazePitch: 0 };

  const leftEyeWidth = distance(lm[33], lm[133]);
  const leftEyeCenter = { x: (lm[33].x + lm[133].x) / 2, y: (lm[33].y + lm[133].y) / 2 };
  const leftGazeYaw = (leftIris.x - leftEyeCenter.x) / leftEyeWidth;
  const leftGazePitch = (leftIris.y - leftEyeCenter.y) / leftEyeWidth;

  const rightEyeWidth = distance(lm[362], lm[263]);
  const rightEyeCenter = { x: (lm[362].x + lm[263].x) / 2, y: (lm[362].y + lm[263].y) / 2 };
  const rightGazeYaw = (rightIris.x - rightEyeCenter.x) / rightEyeWidth;
  const rightGazePitch = (rightIris.y - rightEyeCenter.y) / rightEyeWidth;

  return {
    gazeYaw: (leftGazeYaw + rightGazeYaw) / 2,
    gazePitch: (leftGazePitch + rightGazePitch) / 2,
  };
}

function calculateEMA(current, previous, alpha = 0.35) {
  return alpha * current + (1 - alpha) * previous;
}

// ---------- temporal debounce (ported from proctor-x lib/temporal.ts) ----------

/**
 * Requires a condition to hold for several frames before it counts, and decays
 * faster than it accumulates. One glance away is not a violation; a sustained
 * one is.
 */
class HysteresisBuffer {
  constructor(max, triggerThreshold, decay = 2) {
    this.count = 0;
    this.max = max;
    this.triggerThreshold = triggerThreshold;
    this.decay = decay;
  }

  update(condition) {
    if (condition) {
      this.count = Math.min(this.max, this.count + 1);
    } else {
      this.count = Math.max(0, this.count - this.decay);
    }
    return this.isTriggered;
  }

  get isTriggered() {
    return this.count >= this.triggerThreshold;
  }
}

// ---------- thresholds (proctor-x exam page values) ----------

const YAW_LIMIT = 10;        // degrees before the head counts as turned
const PITCH_LIMIT = 12;      // degrees before it counts as looking down
const GAZE_LIMIT = 0.12;     // iris offset before the eyes count as off-screen
const MAR_TALKING = 0.18;    // mouth openness that may be speech
const MAR_VARIANCE = 0.003;  // movement of the mouth, to separate speech from a yawn
const EAR_CLOSED = 0.15;     // eyes effectively shut

// ---------- engine ----------

export class ProctorXEngine {
  constructor() {
    this.ready = false;
    this.faceLandmarker = null;
    this.handLandmarker = null;
    this.loopId = null;

    this.state = { yawEma: 0, pitchEma: 0, marHistory: [] };

    this.buffers = {
      lookingAway: new HysteresisBuffer(6, 6, 2),
      lookingDown: new HysteresisBuffer(10, 10, 2),
      noFace: new HysteresisBuffer(12, 12, 2),
      multiFace: new HysteresisBuffer(5, 4, 1),
      talking: new HysteresisBuffer(8, 6, 2),
      eyesClosed: new HysteresisBuffer(10, 10, 2),
    };

    this.latest = { flags: [], hands: [], metrics: {} };
  }

  async init() {
    const vision = await import(`${MEDIAPIPE_CDN}/vision_bundle.mjs`);
    const { FilesetResolver, FaceLandmarker, HandLandmarker } = vision;

    const fileset = await FilesetResolver.forVisionTasks(`${MEDIAPIPE_CDN}/wasm`);

    this.faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: `${MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
        delegate: 'CPU',
      },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      runningMode: 'VIDEO',
      numFaces: 2,
    });

    this.handLandmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: `${MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
    });

    this.ready = true;
    return this;
  }

  /** Runs detection at ~15fps against a playing <video>. */
  start(videoEl, fps = 15) {
    this.stop();
    const frameDelay = 1000 / fps;
    let lastRun = 0;

    const loop = (timestamp) => {
      this.loopId = requestAnimationFrame(loop);
      if (timestamp - lastRun < frameDelay) return;
      if (!videoEl || videoEl.readyState < 2 || !videoEl.videoWidth) return;
      lastRun = timestamp;
      try {
        this.processFrame(videoEl);
      } catch (err) {
        console.warn('[proctor-x] frame failed:', err);
      }
    };
    this.loopId = requestAnimationFrame(loop);
  }

  stop() {
    if (this.loopId) cancelAnimationFrame(this.loopId);
    this.loopId = null;
  }

  processFrame(videoEl) {
    const now = performance.now();
    const faces = this.faceLandmarker.detectForVideo(videoEl, now);
    const hands = this.handLandmarker.detectForVideo(videoEl, now);

    const faceCount = faces.faceLandmarks ? faces.faceLandmarks.length : 0;
    let lookingAway = false, lookingDown = false, talking = false, eyesClosed = false;
    const metrics = { faces: faceCount };

    if (faceCount === 1) {
      const lm = faces.faceLandmarks[0];
      const matrix = faces.facialTransformationMatrixes && faces.facialTransformationMatrixes[0];

      if (matrix) {
        const pose = getHeadPose(matrix.data || matrix);
        this.state.yawEma = calculateEMA(pose.yaw, this.state.yawEma);
        this.state.pitchEma = calculateEMA(pose.pitch, this.state.pitchEma);
        metrics.yaw = +this.state.yawEma.toFixed(1);
        metrics.pitch = +this.state.pitchEma.toFixed(1);
      }

      const gaze = getGaze(lm);
      const ear = getEAR(lm);
      const mar = getMAR(lm);
      metrics.gazeYaw = +gaze.gazeYaw.toFixed(3);
      metrics.ear = +ear.toFixed(3);
      metrics.mar = +mar.toFixed(3);

      // Mouth movement over a short window separates speech from a held-open mouth.
      const history = this.state.marHistory;
      history.push(mar);
      if (history.length > 8) history.shift();
      const avg = history.reduce((a, b) => a + b, 0) / history.length;
      const variance = history.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / history.length;

      lookingAway = Math.abs(this.state.yawEma) > YAW_LIMIT || Math.abs(gaze.gazeYaw) > GAZE_LIMIT;
      lookingDown = Math.abs(this.state.pitchEma) > PITCH_LIMIT || Math.abs(gaze.gazePitch) > GAZE_LIMIT;
      talking = mar > MAR_TALKING && variance > MAR_VARIANCE;
      eyesClosed = ear < EAR_CLOSED;
    }

    const b = this.buffers;
    b.lookingAway.update(lookingAway);
    b.lookingDown.update(lookingDown);
    b.noFace.update(faceCount === 0);
    b.multiFace.update(faceCount > 1);
    b.talking.update(talking);
    b.eyesClosed.update(eyesClosed);

    const flags = [];
    if (b.lookingAway.isTriggered) flags.push('LOOKING_AWAY');
    if (b.lookingDown.isTriggered) flags.push('LOOKING_DOWN');
    if (b.noFace.isTriggered) flags.push('NO_FACE');
    if (b.multiFace.isTriggered) flags.push('MULTIPLE_FACES');
    if (b.talking.isTriggered) flags.push('TALKING');
    if (b.eyesClosed.isTriggered) flags.push('EYES_CLOSED');

    // Hand boxes, padded 10%, so the server can tell whether a detected object is
    // actually being held rather than just present somewhere in the room.
    const handBoxes = [];
    if (hands.landmarks) {
      for (const hand of hands.landmarks) {
        let minX = 1, minY = 1, maxX = 0, maxY = 0;
        for (const p of hand) {
          if (p.x < minX) minX = p.x;
          if (p.y < minY) minY = p.y;
          if (p.x > maxX) maxX = p.x;
          if (p.y > maxY) maxY = p.y;
        }
        const w = maxX - minX, h = maxY - minY;
        handBoxes.push({
          minX: Math.max(0, minX - w * 0.1),
          minY: Math.max(0, minY - h * 0.1),
          maxX: Math.min(1, maxX + w * 0.1),
          maxY: Math.min(1, maxY + h * 0.1),
        });
      }
    }

    metrics.hands = handBoxes.length;
    this.latest = { flags, hands: handBoxes, metrics };
  }

  /** Whatever the last processed frame concluded. */
  snapshot() {
    return this.latest;
  }
}
