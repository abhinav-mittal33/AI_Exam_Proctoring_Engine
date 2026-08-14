/*
 * Environment probes: looks for the tooling people actually use to defeat a
 * video proctor, rather than for cheating itself.
 *
 * Every check here runs inside the browser sandbox, so every check here can be
 * defeated by someone willing to edit the page. That is not a reason to skip
 * them - it raises the effort required, and the server cross-checks the claims
 * these produce - but it is the reason none of this is called prevention. Real
 * lockdown needs a native client; see REQUIRE_SEB in the server.
 */

// Virtual cameras present themselves as ordinary webcams with recognisable
// names. This is how a pre-recorded or streamed video gets into an exam.
const VIRTUAL_CAMERA_PATTERNS = [
  'obs', 'virtual', 'manycam', 'snap camera', 'droidcam', 'epoccam',
  'iriun', 'xsplit', 'streamlabs', 'camtwist', 'e2esoft', 'vcam',
  'splitcam', 'youcam', 'fake', 'dummy', 'null', 'v4l2loopback',
];

/** Reads the label of every camera the browser can see. */
export async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((d) => d.kind === 'videoinput')
      .map((d) => ({ label: d.label || '', id: d.deviceId }));
  } catch {
    return [];
  }
}

/**
 * Flags a camera whose name matches known virtual-camera software.
 *
 * Labels are only populated once camera permission has been granted, so this
 * must run after getUserMedia, not before.
 */
export function findVirtualCameras(cameras) {
  return cameras
    .filter((c) => {
      const name = c.label.toLowerCase();
      return VIRTUAL_CAMERA_PATTERNS.some((p) => name.includes(p));
    })
    .map((c) => c.label);
}

/** True when the desktop spans more than one display. */
export function hasSecondDisplay() {
  // Window Management API. Available without a permission prompt in Chromium;
  // undefined elsewhere, in which case we simply do not know.
  if (typeof window.screen.isExtended === 'boolean') return window.screen.isExtended;
  return null;
}

/**
 * Heuristic for an open devtools panel: docked panels shrink the viewport well
 * below the window. Undocked devtools defeat it, so a negative result means
 * nothing - only a positive is worth reporting.
 */
export function devtoolsLikelyOpen() {
  const widthGap = window.outerWidth - window.innerWidth;
  const heightGap = window.outerHeight - window.innerHeight;
  return widthGap > 200 || heightGap > 220;
}

/**
 * One pass over the environment.
 *
 * `activeCameraLabel` is the track actually feeding the exam, which matters more
 * than what is merely installed: a virtual camera sitting unused is untidy, one
 * that is selected is the exam being fed synthetic video.
 */
export async function probeEnvironment(activeCameraLabel = '') {
  const flags = [];
  const detail = {};

  const cameras = await listCameras();
  const virtual = findVirtualCameras(cameras);
  detail.cameras = cameras.map((c) => c.label).filter(Boolean);

  if (virtual.length) {
    detail.virtualCameras = virtual;
    const active = activeCameraLabel.toLowerCase();
    const activeIsVirtual = VIRTUAL_CAMERA_PATTERNS.some((p) => active.includes(p));
    flags.push(activeIsVirtual ? 'VIRTUAL_CAMERA_ACTIVE' : 'VIRTUAL_CAMERA_PRESENT');
  }

  const extended = hasSecondDisplay();
  if (extended === true) {
    flags.push('SECOND_DISPLAY');
    detail.displays = 'extended';
  }

  if (devtoolsLikelyOpen()) {
    flags.push('DEVTOOLS_OPEN');
  }

  detail.screen = `${window.screen.width}x${window.screen.height}`;
  return { flags, detail };
}
