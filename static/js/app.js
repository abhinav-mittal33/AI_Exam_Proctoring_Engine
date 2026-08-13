/*
 * Client for the proctored exam app.
 *
 * Students upload a JPEG every few seconds; that single stream of frames both
 * feeds the cheating detector and paints the host's monitoring grid, so the
 * normal path needs no WebRTC at all. A peer connection is negotiated only when
 * the host opens one candidate for a live look.
 */

const socket = io();

const $ = (id) => document.getElementById(id);

const screens = ['landing', 'register', 'signin', 'exam', 'host', 'ended'];
function show(name) {
  screens.forEach((s) => $('screen-' + s).classList.toggle('hidden', s !== name));
}

// ---------- state ----------
let camStream = null;        // camera feed, shared by preview and exam
let role = null;             // 'host' | 'student'
let examCode = null;
let frameTimer = null;
let livePeer = null;         // host side: connection to the student being viewed
let studentPeer = null;      // student side: connection to the host
let liveStudentId = null;
let proctorX = null;        // browser-side landmark engine (proctor-x port)
let iceConfig = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };
const tiles = {};            // student_id -> tile element
const sessionFlags = new Set();
let audioCtx = null;
let audioSource = null;
let audioAnalyser = null;
let audioTimer = null;
const AUDIO_THRESHOLD = 0.08;

// ---------- camera ----------

async function startCamera(videoEl) {
  if (!camStream) {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: true,
    });
  }
  if (videoEl) {
    videoEl.srcObject = camStream;
    await videoEl.play().catch(() => {});
  }
  return camStream;
}

function stopCamera() {
  if (camStream) {
    camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
  }
}

/** Grabs a JPEG from a playing video element, or null if it has no frame yet. */
function snapshot(videoEl, quality = 0.7) {
  if (!videoEl || !videoEl.videoWidth) return null;
  const canvas = document.createElement('canvas');
  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  canvas.getContext('2d').drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', quality);
}

// ---------- landing ----------

$('btnStartExam').addEventListener('click', async () => {
  $('landingError').textContent = '';
  const btn = $('btnStartExam');
  btn.disabled = true;

  socket.emit('host-start-exam', { title: 'Examination' }, (res) => {
    btn.disabled = false;
    if (!res.ok) { $('landingError').textContent = res.error || 'Could not start the exam.'; return; }

    role = 'host';
    examCode = res.code;
    $('hostExamTitle').textContent = res.title;
    $('hostCodeLabel').textContent = res.code;
    $('emptyLink').textContent = joinLink();
    $('studentCount').textContent = '0';

    const det = res.detectors || {};
    const pill = $('detectorPill');
    if (det.object_detection) {
      pill.className = 'pill pill-ok';
      pill.textContent = 'Detection ready';
    } else {
      // Never let a dead detector look healthy.
      pill.className = 'pill pill-warn';
      pill.textContent = 'Object detection OFF';
    }

    show('host');
  });
});

$('btnJoinExam').addEventListener('click', () => enterSignin($('joinCode').value));
$('joinCode').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') enterSignin($('joinCode').value);
});

async function enterSignin(code) {
  code = (code || '').trim().toUpperCase();
  if (!code) { $('landingError').textContent = 'Enter the exam code first.'; return; }

  examCode = code;
  $('signinCode').textContent = code;
  $('signinMsg').textContent = '';
  show('signin');

  try {
    await startCamera($('signinCam'));
  } catch (err) {
    $('signinMsg').textContent = 'Camera access is required to join an exam.';
    $('signinMsg').className = 'status-text err';
  }
}

$('linkRegister').addEventListener('click', async (e) => {
  e.preventDefault();
  $('registerMsg').textContent = '';
  show('register');
  try {
    await startCamera($('registerCam'));
  } catch (err) {
    $('registerMsg').textContent = 'Camera access is required to register.';
    $('registerMsg').className = 'status-text err';
  }
});

document.querySelectorAll('[data-back]').forEach((btn) => {
  btn.addEventListener('click', () => {
    stopCamera();
    show('landing');
  });
});

// ---------- registration ----------

$('btnRegister').addEventListener('click', async () => {
  const msg = $('registerMsg');
  const image = snapshot($('registerCam'), 0.9);
  if (!image) { msg.className = 'status-text err'; msg.textContent = 'Camera is not ready yet.'; return; }

  const btn = $('btnRegister');
  btn.disabled = true;
  msg.className = 'status-text';
  msg.textContent = 'Checking photo quality…';

  try {
    const res = await fetch('/api/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: $('regName').value.trim(),
        enrollment_number: $('regEnrollment').value.trim(),
        password: $('regPassword').value,
        image_data: image,
      }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Registration failed.');

    msg.className = 'status-text ok';
    msg.textContent = data.message + ' You can now join an exam.';
  } catch (err) {
    msg.className = 'status-text err';
    msg.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

// ---------- student sign-in ----------

$('btnSignin').addEventListener('click', () => {
  const msg = $('signinMsg');
  const image = snapshot($('signinCam'), 0.9);
  if (!image) { msg.className = 'status-text err'; msg.textContent = 'Camera is not ready yet.'; return; }

  const btn = $('btnSignin');
  btn.disabled = true;
  msg.className = 'status-text';
  msg.textContent = 'Verifying your identity…';

  socket.emit('student-join', {
    code: examCode,
    enrollment_number: $('signinEnrollment').value.trim(),
    password: $('signinPassword').value,
    image_data: image,
  }, async (res) => {
    btn.disabled = false;
    if (!res.ok) {
      msg.className = 'status-text err';
      msg.textContent = res.error;
      return;
    }

    role = 'student';
    $('examTitle').textContent = res.title;
    $('examCodeLabel').textContent = res.code;
    $('selfName').textContent = res.name + ' (You)';
    show('exam');

    // Attempt to enter fullscreen
    await enterFullscreen();

    await startCamera($('selfCam'));
    startProctorX(); // Start client-side proctoring engine asynchronously
    startAudioAnalysis(); // Start local audio analysis
    startFrameUploads(res.frame_interval || 3);
  });
});

// ---------- student: frame uploads ----------

async function startProctorX() {
  try {
    const { ProctorXEngine } = await import('/static/js/proctorx.js');
    const engine = new ProctorXEngine();
    await engine.init();
    engine.start($('selfCam'), 15);
    proctorX = engine;
    console.log('[proctor-x] landmark engine running');
  } catch (err) {
    // The server still runs object and face-count detection on the uploaded
    // frames, so proctoring degrades rather than stops.
    console.warn('[proctor-x] could not start; server-side detection continues:', err);
    proctorX = null;
  }
}

function startFrameUploads(intervalSecs) {
  stopFrameUploads();
  const send = () => {
    const image = snapshot($('selfCam'), 0.7);
    // No frame means the camera is off or not ready. The server notices the gap
    // and tells the proctor, rather than detection silently doing nothing.
    if (!image) return;

    const payload = { image_data: image };
    
    // Gather and clear active event-driven flags
    let currentFlags = Array.from(sessionFlags);
    sessionFlags.clear();

    // If they are not in fullscreen, keep adding the exit flag
    if (!document.fullscreenElement) {
      sessionFlags.add('EXIT_FULLSCREEN');
      if (!currentFlags.includes('EXIT_FULLSCREEN')) {
        currentFlags.push('EXIT_FULLSCREEN');
      }
    }

    if (proctorX) {
      const { flags, hands, metrics } = proctorX.snapshot();
      currentFlags = currentFlags.concat(flags);
      payload.hands = hands;    // so the server can tell "held" from "in the room"
      payload.metrics = metrics;
    }
    payload.flags = currentFlags;
    payload.client_landmarks_active = !!proctorX;
    socket.emit('student-frame', payload);
  };
  send();
  frameTimer = setInterval(send, intervalSecs * 1000);
}

function triggerImmediateUpload() {
  if (role === 'student' && camStream) {
    const image = snapshot($('selfCam'), 0.7);
    if (!image) return;
    const payload = { image_data: image };
    let currentFlags = Array.from(sessionFlags);
    sessionFlags.clear();

    if (!document.fullscreenElement) {
      sessionFlags.add('EXIT_FULLSCREEN');
      if (!currentFlags.includes('EXIT_FULLSCREEN')) {
        currentFlags.push('EXIT_FULLSCREEN');
      }
    }

    if (proctorX) {
      const { flags, hands, metrics } = proctorX.snapshot();
      currentFlags = currentFlags.concat(flags);
      payload.hands = hands;
      payload.metrics = metrics;
    }
    payload.flags = currentFlags;
    payload.client_landmarks_active = !!proctorX;
    socket.emit('student-frame', payload);
  }
}

function stopFrameUploads() {
  if (frameTimer) clearInterval(frameTimer);
  frameTimer = null;
  if (proctorX) { proctorX.stop(); proctorX = null; }
}

socket.on('proctor-status', ({ warning_count }) => {
  $('warnCount').textContent = warning_count;
});

socket.on('proctor-warning', ({ warning_count, reasons, removed }) => {
  $('warnCount').textContent = warning_count;

  const badge = $('selfWarnBadge');
  badge.textContent = `Warning ${warning_count}/3`;
  badge.classList.remove('hidden');

  const toast = $('warningToast');
  toast.textContent = removed
    ? 'Removed from the exam: ' + reasons.join(', ')
    : `Warning ${warning_count} of 3 — ${reasons.join(', ')}`;
  toast.classList.remove('hidden');
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.add('hidden'), 6000);

  const pill = $('proctorPill');
  pill.className = 'pill pill-warn';
  pill.textContent = `${warning_count} warning${warning_count === 1 ? '' : 's'}`;
});

$('btnLeaveExam').addEventListener('click', () => {
  exitFullscreen();
  stopAudioAnalysis();
  stopFrameUploads();
  stopCamera();
  socket.disconnect();
  $('endedReason').textContent = 'You left the exam.';
  show('ended');
  setTimeout(() => socket.connect(), 500);
});

$('btnMic').addEventListener('click', () => {
  if (!camStream) return;
  const track = camStream.getAudioTracks()[0];
  if (!track) return;
  track.enabled = !track.enabled;
  $('btnMic').classList.toggle('off', !track.enabled);
});

$('btnCam').addEventListener('click', () => {
  if (!camStream) return;
  const track = camStream.getVideoTracks()[0];
  if (!track) return;
  track.enabled = !track.enabled;
  $('btnCam').classList.toggle('off', !track.enabled);
});

// ---------- host: monitoring grid ----------

function joinLink() {
  return `${location.origin}/?code=${examCode}`;
}

$('btnCopyLink').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(joinLink());
    $('btnCopyLink').textContent = 'Copied';
    setTimeout(() => ($('btnCopyLink').textContent = 'Copy join link'), 1500);
  } catch (err) {
    $('btnCopyLink').textContent = joinLink();
  }
});

function ensureTile(studentId, name) {
  if (tiles[studentId]) return tiles[studentId];

  $('emptyState').classList.add('hidden');

  const tile = document.createElement('div');
  tile.className = 'student-tile';
  tile.id = 'tile-' + studentId;
  tile.innerHTML = `
    <div class="no-signal">Waiting for video…</div>
    <span class="tile-label">${escapeHtml(name)}</span>
    <span class="tile-live-hint">Click for live view</span>
  `;
  tile.addEventListener('click', () => openLiveView(studentId, name));

  $('hostGrid').appendChild(tile);
  tiles[studentId] = tile;
  updateStudentCount();
  return tile;
}

function updateStudentCount() {
  const n = Object.keys(tiles).length;
  $('studentCount').textContent = n;
  $('studentCountLabel').textContent = n === 1 ? 'candidate' : 'candidates';
  if (n === 0) $('emptyState').classList.remove('hidden');
}

socket.on('student-joined', ({ student_id, name, enrollment }) => {
  ensureTile(student_id, `${name} · ${enrollment}`);
  addAlert(name, 'Joined the exam — identity verified', 'info');
});

socket.on('student-frame', ({ student_id, name, enrollment, image_data }) => {
  const tile = ensureTile(student_id, `${name} · ${enrollment}`);
  tile.classList.remove('stalled');

  let img = tile.querySelector('img');
  if (!img) {
    img = document.createElement('img');
    tile.insertBefore(img, tile.firstChild);
    const placeholder = tile.querySelector('.no-signal');
    if (placeholder) placeholder.remove();
  }
  img.src = image_data;
});

socket.on('proctor-alert', ({ student_id, name, warning_count, reasons, removed }) => {
  const tile = tiles[student_id];
  if (tile) {
    tile.classList.add('flagged');
    let flag = tile.querySelector('.tile-flag');
    if (!flag) {
      flag = document.createElement('span');
      flag.className = 'tile-flag';
      tile.appendChild(flag);
    }
    flag.textContent = removed ? 'Removed' : `Warning ${warning_count}/3`;
  }
  addAlert(name, reasons.join(' • ') + (removed ? ' — removed from exam' : ''));
});

socket.on('proctor-stalled', ({ student_id, name, stalled_seconds }) => {
  const tile = tiles[student_id];
  if (tile) tile.classList.add('stalled');
  addAlert(name, `No camera frames for ${stalled_seconds}s — monitoring may be blocked`, 'info');
});

socket.on('student-left', ({ student_id, name, reason }) => {
  const tile = tiles[student_id];
  if (tile) tile.remove();
  delete tiles[student_id];
  updateStudentCount();
  addAlert(name, reason, 'info');
  if (liveStudentId === student_id) closeLiveView();
});

function addAlert(name, text, kind = 'violation') {
  const empty = $('alertFeed').querySelector('.feed-empty');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'alert-item' + (kind === 'info' ? ' info' : '');
  item.innerHTML = `
    <div class="alert-head">
      <span>${escapeHtml(name)}</span>
      <span class="alert-time">${new Date().toLocaleTimeString()}</span>
    </div>
    <div class="alert-reason">${escapeHtml(text)}</div>
  `;
  $('alertFeed').prepend(item);
}

$('btnEndExam').addEventListener('click', () => {
  socket.emit('host-end-exam', {}, () => {
    $('endedReason').textContent = 'You ended the exam for everyone.';
    show('ended');
  });
});

// ---------- live view (WebRTC, host <-> one student) ----------

fetch('/api/ice-config')
  .then((r) => r.json())
  .then((cfg) => {
    if (cfg.iceServers && cfg.iceServers.length) iceConfig = { iceServers: cfg.iceServers };
    if (!cfg.relayReady) {
      console.warn('[live view] No TURN relay configured. The monitoring grid is ' +
                   'unaffected, but a live view may fail across networks.');
    }
  })
  .catch(() => {});

function openLiveView(studentId, name) {
  liveStudentId = studentId;
  $('liveName').textContent = 'Live view — ' + name;
  $('liveStatus').textContent = 'Connecting…';
  $('liveModal').classList.remove('hidden');

  socket.emit('live-view-request', { student_id: studentId }, (res) => {
    if (!res.ok) $('liveStatus').textContent = res.error;
  });
}

function closeLiveView() {
  $('liveModal').classList.add('hidden');
  $('liveVideo').srcObject = null;
  if (livePeer) { livePeer.close(); livePeer = null; }
  if (liveStudentId) socket.emit('live-view-stop', { student_id: liveStudentId });
  liveStudentId = null;
}

$('btnCloseLive').addEventListener('click', closeLiveView);

// Student side: the host asked to watch, so publish the camera to them.
socket.on('live-view-request', async ({ host_id }) => {
  if (studentPeer) studentPeer.close();
  studentPeer = new RTCPeerConnection(iceConfig);

  camStream.getTracks().forEach((track) => studentPeer.addTrack(track, camStream));

  studentPeer.onicecandidate = (e) => {
    if (e.candidate) {
      socket.emit('signal', { to: host_id, data: { type: 'candidate', candidate: e.candidate } });
    }
  };

  const offer = await studentPeer.createOffer();
  await studentPeer.setLocalDescription(offer);
  socket.emit('signal', { to: host_id, data: { type: 'offer', sdp: offer } });
});

socket.on('live-view-stop', () => {
  if (studentPeer) { studentPeer.close(); studentPeer = null; }
});

socket.on('signal', async ({ from, data }) => {
  if (role === 'host') {
    if (data.type === 'offer') {
      livePeer = new RTCPeerConnection(iceConfig);

      livePeer.ontrack = (e) => {
        $('liveVideo').srcObject = e.streams[0];
        $('liveVideo').play().catch((err) => {
          // Surfaced rather than swallowed: a blocked play() looks identical to
          // a failed connection otherwise.
          $('liveStatus').textContent = 'Click the video to start playback.';
          console.warn('[live view] play() rejected:', err);
        });
        $('liveStatus').textContent = 'Live';
      };

      livePeer.onicecandidate = (e) => {
        if (e.candidate) {
          socket.emit('signal', { to: from, data: { type: 'candidate', candidate: e.candidate } });
        }
      };

      livePeer.onconnectionstatechange = () => {
        const state = livePeer.connectionState;
        console.log('[live view] connectionState:', state);
        if (state === 'failed') {
          $('liveStatus').textContent =
            'Could not connect a live stream (no TURN relay). The grid still updates.';
        }
      };

      await livePeer.setRemoteDescription(new RTCSessionDescription(data.sdp));
      const answer = await livePeer.createAnswer();
      await livePeer.setLocalDescription(answer);
      socket.emit('signal', { to: from, data: { type: 'answer', sdp: answer } });

    } else if (data.type === 'candidate' && livePeer) {
      await livePeer.addIceCandidate(new RTCIceCandidate(data.candidate)).catch(() => {});
    }

  } else if (role === 'student' && studentPeer) {
    if (data.type === 'answer') {
      await studentPeer.setRemoteDescription(new RTCSessionDescription(data.sdp));
    } else if (data.type === 'candidate') {
      await studentPeer.addIceCandidate(new RTCIceCandidate(data.candidate)).catch(() => {});
    }
  }
});

// ---------- exam end ----------

socket.on('exam-ended', ({ reason }) => {
  exitFullscreen();
  stopAudioAnalysis();
  stopFrameUploads();
  stopCamera();
  if (studentPeer) { studentPeer.close(); studentPeer = null; }
  $('endedReason').textContent = reason || 'This exam session is over.';
  show('ended');
});

// ---------- helpers ----------

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str == null ? '' : String(str);
  return div.innerHTML;
}

async function enterFullscreen() {
  const elem = document.documentElement;
  try {
    if (elem.requestFullscreen) {
      await elem.requestFullscreen();
    } else if (elem.webkitRequestFullscreen) {
      await elem.webkitRequestFullscreen();
    } else if (elem.msRequestFullscreen) {
      await elem.msRequestFullscreen();
    }
  } catch (err) {
    console.error('Error entering fullscreen:', err);
  }
}

function exitFullscreen() {
  if (document.fullscreenElement) {
    if (document.exitFullscreen) {
      document.exitFullscreen().catch(() => {});
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen().catch(() => {});
    } else if (document.msExitFullscreen) {
      document.msExitFullscreen().catch(() => {});
    }
  }
}

function setupProctorRestrictions() {
  const handleFullscreenChange = () => {
    if (role !== 'student') return;
    if (!document.fullscreenElement) {
      $('fullscreenBlocker').classList.remove('hidden');
      sessionFlags.add('EXIT_FULLSCREEN');
      triggerImmediateUpload();
    } else {
      $('fullscreenBlocker').classList.add('hidden');
    }
  };

  document.addEventListener('fullscreenchange', handleFullscreenChange);
  document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
  document.addEventListener('msfullscreenchange', handleFullscreenChange);

  document.addEventListener('visibilitychange', () => {
    if (role !== 'student') return;
    if (document.visibilityState === 'hidden') {
      sessionFlags.add('TAB_SWITCH');
      triggerImmediateUpload();
    }
  });

  window.addEventListener('blur', () => {
    if (role !== 'student') return;
    sessionFlags.add('TAB_SWITCH');
    triggerImmediateUpload();
  });

  $('btnReenterFullscreen').addEventListener('click', async () => {
    await enterFullscreen();
  });

  // Intercept right-clicks (context menu)
  document.addEventListener('contextmenu', (e) => {
    if (role !== 'student') return;
    e.preventDefault();
    sessionFlags.add('SHORTCUT_ATTEMPT');
    triggerImmediateUpload();
  });

  // Intercept Copy, Cut, Paste
  document.addEventListener('copy', (e) => {
    if (role !== 'student') return;
    e.preventDefault();
    sessionFlags.add('SHORTCUT_ATTEMPT');
    triggerImmediateUpload();
  });

  document.addEventListener('cut', (e) => {
    if (role !== 'student') return;
    e.preventDefault();
    sessionFlags.add('SHORTCUT_ATTEMPT');
    triggerImmediateUpload();
  });

  document.addEventListener('paste', (e) => {
    if (role !== 'student') return;
    e.preventDefault();
    sessionFlags.add('SHORTCUT_ATTEMPT');
    triggerImmediateUpload();
  });

  // Intercept developer tools and system shortcuts
  document.addEventListener('keydown', (e) => {
    if (role !== 'student') return;
    const isCmdOrCtrl = e.metaKey || e.ctrlKey;
    const key = e.key.toLowerCase();
    
    if (
      (isCmdOrCtrl && (key === 'c' || key === 'v' || key === 'x' || key === 'u')) ||
      (isCmdOrCtrl && e.shiftKey && key === 'i') ||
      e.key === 'F12'
    ) {
      e.preventDefault();
      sessionFlags.add('SHORTCUT_ATTEMPT');
      triggerImmediateUpload();
    }
  });
}

function startAudioAnalysis() {
  stopAudioAnalysis();
  if (!camStream) return;
  
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;
    
    audioCtx = new AudioContextClass();
    audioSource = audioCtx.createMediaStreamSource(camStream);
    audioAnalyser = audioCtx.createAnalyser();
    audioAnalyser.fftSize = 256;
    audioSource.connect(audioAnalyser);
    
    const bufferLength = audioAnalyser.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);
    
    let voiceFramesCount = 0;
    
    audioTimer = setInterval(() => {
      if (role !== 'student' || !audioAnalyser) return;
      audioAnalyser.getFloatTimeDomainData(dataArray);
      
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sum / bufferLength);
      
      if (rms > AUDIO_THRESHOLD) {
        voiceFramesCount++;
      } else {
        voiceFramesCount = Math.max(0, voiceFramesCount - 1);
      }
      
      if (voiceFramesCount >= 5) {
        sessionFlags.add('AUDIO_TALKING');
        triggerImmediateUpload();
        voiceFramesCount = 0;
      }
    }, 200);
    
  } catch (err) {
    console.warn('[audio-analysis] could not start:', err);
  }
}

function stopAudioAnalysis() {
  if (audioTimer) {
    clearInterval(audioTimer);
    audioTimer = null;
  }
  if (audioSource) {
    audioSource.disconnect();
    audioSource = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }
  audioAnalyser = null;
}

// Initialize proctor restrictions listener
setupProctorRestrictions();

// A shared join link lands here with ?code=ABC123 already filled in.
const codeFromUrl = new URLSearchParams(location.search).get('code');
if (codeFromUrl) {
  $('joinCode').value = codeFromUrl.toUpperCase();
}
