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

// ---------- thresholds ----------
//
// Pose limits are deviations from the candidate's own calibrated resting
// posture, not absolute angles. Absolute limits punish anyone who naturally
// sits at an angle or has their webcam mounted off-centre, which is where most
// of the false positives came from.

const CALIBRATION_FRAMES = 45;   // ~3s at 15fps to learn a resting posture
const YAW_LIMIT = 22;            // degrees off resting before the head counts as turned
const PITCH_LIMIT = 18;          // degrees off resting before it counts as looking down
const ROLL_LIMIT = 20;           // degrees off resting before the head counts as tilted
const GAZE_LIMIT = 0.16;         // iris offset before the eyes count as off-screen
const EAR_CLOSED_RATIO = 0.62;   // fraction of resting eye opening that counts as shut

// Speech is rhythmic: the jaw opens and closes repeatedly. Counting those cycles
// catches quiet talking that never opens the mouth far, while ignoring a single
// yawn or a mouth simply held open - both of which are one cycle, not several.
const MAR_SPEECH_DELTA = 0.030;  // opening above resting that counts as "open"
const SPEECH_WINDOW_MS = 2500;   // window over which cycles are counted
const SPEECH_MIN_CYCLES = 3;     // open/close cycles in that window to call it speech

// ---------- engine ----------

export class ProctorXEngine {
  constructor() {
    this.ready = false;
    this.faceLandmarker = null;
    this.handLandmarker = null;
    this.loopId = null;

    this.state = {
      yawEma: 0, pitchEma: 0, rollEma: 0,
      marSamples: [],     // {t, mar} over the speech window
      mouthOpen: false,   // for counting open/close cycles
      gazeDirection: null, // LEFT / RIGHT / UP / DOWN when eyes leave the screen
      cycles: [],         // timestamps of mouth openings
    };

    // Resting posture, learned in the first few seconds and then used as the
    // origin every pose measurement is compared against.
    this.calibration = { samples: [], done: false, neutral: null };

    // Frames each signal must hold before it counts. Serious, unambiguous things
    // (a second person) fire fast; subjective ones need more evidence.
    this.buffers = {
      lookingAway: new HysteresisBuffer(14, 12, 3),
      lookingDown: new HysteresisBuffer(16, 14, 3),
      headTilt: new HysteresisBuffer(16, 14, 3),
      gazeOff: new HysteresisBuffer(16, 14, 3),
      noFace: new HysteresisBuffer(12, 10, 2),
      multiFace: new HysteresisBuffer(5, 3, 1),
      talking: new HysteresisBuffer(6, 4, 2),
      eyesClosed: new HysteresisBuffer(20, 18, 2),
    };

    this.latest = { flags: [], hands: [], metrics: {}, calibrating: true };
  }

  /** Median is used over mean so a stray frame cannot skew the resting pose. */
  static median(values) {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
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
      // Look for more than two people, and lower the bar for spotting them: a
      // second person leaning into frame is usually partly cut off or turned
      // away, and at the default confidence they were being missed entirely.
      numFaces: 4,
      minFaceDetectionConfidence: 0.35,
      minFacePresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
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
    let lookingAway = false, lookingDown = false, headTilt = false;
    let gazeOff = false, talking = false, eyesClosed = false;
    const metrics = { faces: faceCount };

    if (faceCount >= 1) {
      const lm = faces.faceLandmarks[0];
      const matrix = faces.facialTransformationMatrixes && faces.facialTransformationMatrixes[0];

      let yaw = 0, pitch = 0, roll = 0;
      if (matrix) {
        const pose = getHeadPose(matrix.data || matrix);
        this.state.yawEma = calculateEMA(pose.yaw, this.state.yawEma);
        this.state.pitchEma = calculateEMA(pose.pitch, this.state.pitchEma);
        this.state.rollEma = calculateEMA(pose.roll, this.state.rollEma);
        yaw = this.state.yawEma;
        pitch = this.state.pitchEma;
        roll = this.state.rollEma;
      }

      const gaze = getGaze(lm);
      const ear = getEAR(lm);
      const mar = getMAR(lm);

      // --- calibration: learn this candidate's resting posture first ---
      if (!this.calibration.done) {
        this.calibration.samples.push({
          yaw, pitch, roll, mar, ear,
          gazeYaw: gaze.gazeYaw, gazePitch: gaze.gazePitch,
        });
        if (this.calibration.samples.length >= CALIBRATION_FRAMES) {
          const s = this.calibration.samples;
          const pick = (key) => ProctorXEngine.median(s.map((x) => x[key]));
          this.calibration.neutral = {
            yaw: pick('yaw'), pitch: pick('pitch'), roll: pick('roll'),
            mar: pick('mar'), ear: pick('ear'),
            // Resting iris position. A webcam above or beside the screen means a
            // candidate looking straight at their work already has a non-zero
            // offset, so gaze must be measured from their own centre too.
            gazeYaw: pick('gazeYaw'), gazePitch: pick('gazePitch'),
          };
          this.calibration.done = true;
          console.log('[proctor-x] calibrated resting posture:', this.calibration.neutral);
        }
        // Face count still matters during calibration - a second person present
        // from the start is exactly the case worth catching.
        this.buffers.noFace.update(false);
        this.buffers.multiFace.update(faceCount > 1);
        this.latest = {
          flags: this.buffers.multiFace.isTriggered ? ['MULTIPLE_FACES'] : [],
          hands: [],
          metrics: { ...metrics, calibrating: true },
          calibrating: true,
        };
        return;
      }

      const n = this.calibration.neutral;
      const yawDev = Math.abs(yaw - n.yaw);
      const pitchDev = Math.abs(pitch - n.pitch);
      const rollDev = Math.abs(roll - n.roll);

      // Iris offset measured from this candidate's own resting gaze, so an
      // off-centre webcam does not read as permanently looking away.
      const gazeYawDev = gaze.gazeYaw - n.gazeYaw;
      const gazePitchDev = gaze.gazePitch - n.gazePitch;

      metrics.yawDev = +yawDev.toFixed(1);
      metrics.pitchDev = +pitchDev.toFixed(1);
      metrics.rollDev = +rollDev.toFixed(1);
      metrics.gazeYawDev = +gazeYawDev.toFixed(3);
      metrics.gazePitchDev = +gazePitchDev.toFixed(3);
      metrics.ear = +ear.toFixed(3);
      metrics.mar = +mar.toFixed(3);

      lookingAway = yawDev > YAW_LIMIT;
      lookingDown = pitchDev > PITCH_LIMIT;
      headTilt = rollDev > ROLL_LIMIT;
      eyesClosed = ear < n.ear * EAR_CLOSED_RATIO;

      // Eyes can leave the screen while the head stays perfectly still, which is
      // exactly how someone reads from a second screen or a note beside them, so
      // gaze is tracked as its own signal rather than as a corollary of pose.
      // Ignored while the eyes are shut, since iris position is meaningless then.
      if (!eyesClosed) {
        if (Math.abs(gazeYawDev) > GAZE_LIMIT) {
          gazeOff = true;
          // The preview is mirrored for the candidate, but landmark x grows to
          // the image right, which is the candidate's own left.
          this.state.gazeDirection = gazeYawDev > 0 ? 'LEFT' : 'RIGHT';
        } else if (Math.abs(gazePitchDev) > GAZE_LIMIT) {
          gazeOff = true;
          this.state.gazeDirection = gazePitchDev > 0 ? 'DOWN' : 'UP';
        }
      }
      metrics.gazeDirection = gazeOff ? this.state.gazeDirection : null;

      // --- speech: count open/close cycles rather than raw openness ---
      // A mouth held open, or one yawn, is a single cycle. Talking is several.
      const openNow = mar > n.mar + MAR_SPEECH_DELTA;
      if (openNow && !this.state.mouthOpen) this.state.cycles.push(now);
      this.state.mouthOpen = openNow;
      this.state.cycles = this.state.cycles.filter((t) => now - t <= SPEECH_WINDOW_MS);
      talking = this.state.cycles.length >= SPEECH_MIN_CYCLES;
      metrics.speechCycles = this.state.cycles.length;
    }

    const b = this.buffers;
    b.lookingAway.update(lookingAway);
    b.lookingDown.update(lookingDown);
    b.headTilt.update(headTilt);
    b.gazeOff.update(gazeOff);
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
    if (b.headTilt.isTriggered) flags.push('HEAD_TILT');

    // Gaze is reported on its own axis. Suppressed only when the head is already
    // turned the same way - that is one act, and head-turn already covers it -
    // but eyes drifting sideways while the head stays still is its own finding.
    if (b.gazeOff.isTriggered) {
      const dir = this.state.gazeDirection;
      const sameAxisAsHead =
        ((dir === 'LEFT' || dir === 'RIGHT') && lookingAway) ||
        ((dir === 'UP' || dir === 'DOWN') && lookingDown);
      if (!sameAxisAsHead) flags.push('GAZE_' + dir);
    }

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
    this.latest = { flags, hands: handBoxes, metrics, calibrating: false };
  }

  /** True until the resting posture has been learned. */
  get isCalibrating() {
    return !this.calibration.done;
  }

  /** Whatever the last processed frame concluded. */
  snapshot() {
    return this.latest;
  }
}
