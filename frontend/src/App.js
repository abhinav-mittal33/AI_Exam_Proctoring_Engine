import React, { useState, useEffect, useRef } from 'react';
import Webcam from 'react-webcam';
import {
  ThemeProvider, createTheme, CssBaseline, Container, Grid, Paper, Typography, Box,
  Chip, LinearProgress, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Tabs, Tab,
  Dialog, DialogTitle, DialogContent, DialogActions, Button, Divider, Alert
} from '@mui/material';
import {
  Videocam, Security, Shield, Speed, FilterList, Person, Info, Analytics, Lightbulb, PlayArrow, CheckCircle, Warning, Timer, Psychology, AutoFixHigh
} from '@mui/icons-material';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    background: { default: '#0b0f19', paper: '#141c2e' },
    primary: { main: '#06b6d4' },
    secondary: { main: '#3b82f6' }
  },
  typography: { fontFamily: 'Inter, system-ui, sans-serif' }
});

const API_HOST = "http://localhost:8001";
const WS_HOST = "ws://localhost:8001";

export default function App() {
  const [sessionId] = useState("session_" + Math.random().toString(36).substring(2, 9));
  const [studentName] = useState("Abhinav Mittal");
  const [enrollment] = useState("23CSE10452");

  const [isExamStarted, setIsExamStarted] = useState(false);
  const [graceRemaining, setGraceRemaining] = useState(5.0);
  const [isInGracePeriod, setIsInGracePeriod] = useState(false);

  const [connected, setConnected] = useState(false);
  const [frameQuality, setFrameQuality] = useState("GOOD");
  const [riskScore, setRiskScore] = useState(0.0);
  const [riskLevel, setRiskLevel] = useState("NORMAL");
  const [riskBreakdown, setRiskBreakdown] = useState({
    category_scores: { OBJECT: 0.0, FACE: 0.0, GAZE: 0.0, HEAD: 0.0, MOUTH: 0.0, IDENTITY: 0.0 },
    decay_adjustment: 0.0,
    correlation_adjustment: 0.0
  });

  const [mlPrediction, setMlPrediction] = useState({
    top_behavior: "NORMAL",
    confidence_score: 0.95,
    probabilities: { NORMAL: 0.95, PHONE_USE: 0.01, MULTIPLE_PERSON: 0.01, FACE_ABSENT: 0.01, PERSISTENT_GAZE_AWAY: 0.01, PERSISTENT_HEAD_TURN: 0.01 },
    model_version: "XGBoost Temporal Classifier v1.1.0"
  });

  const [temporalFeatures, setTemporalFeatures] = useState({});

  const [currentStatus, setCurrentStatus] = useState({
    face_count: 0,
    face_state: "NO_FACE",
    distance_state: "NO_FACE",
    centering_state: "NO_FACE",
    face_width_pct: 0.0,
    alignment_passed: false,
    head_direction: "CENTER",
    yaw: 0.0,
    pitch: 0.0,
    gaze_direction: "GAZE_CENTER",
    gaze_ratio: 0.50,
    mouth_state: "NORMAL",
    mar: 0.05,
    prohibited_objects: []
  });

  const [events, setEvents] = useState([]);
  const [filterCategory, setFilterCategory] = useState("ALL");
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [activeTab, setActiveTab] = useState("TIMELINE");

  const webcamRef = useRef(null);
  const wsRef = useRef(null);

  const isAligned = currentStatus.alignment_passed || (
    currentStatus.face_count === 1 &&
    currentStatus.distance_state === 'OPTIMAL' &&
    currentStatus.centering_state === 'CENTERED'
  );

  const handleStartExam = () => {
    setIsExamStarted(true);
    fetch(`${API_HOST}/api/proctor/session/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, student_id: enrollment, exam_id: "EXAM_2026", student_name: studentName })
    }).catch(console.error);
  };

  useEffect(() => {
    const ws = new WebSocket(`${WS_HOST}/ws/proctor/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data);
        if (data.current_status) setCurrentStatus(data.current_status);
        if (data.frame_quality) setFrameQuality(data.frame_quality);
        if (data.is_in_grace_period !== undefined) setIsInGracePeriod(data.is_in_grace_period);
        if (data.grace_seconds_remaining !== undefined) setGraceRemaining(data.grace_seconds_remaining);
        if (data.risk_score !== undefined) setRiskScore(data.risk_score);
        if (data.risk_level) setRiskLevel(data.risk_level);
        if (data.risk_breakdown) setRiskBreakdown(data.risk_breakdown);
        if (data.ml_prediction) setMlPrediction(data.ml_prediction);
        if (data.temporal_features) setTemporalFeatures(data.temporal_features);

        if (data.new_events && data.new_events.length > 0) {
          setEvents(prev => {
            const existingIds = new Set(prev.map(e => e.event_id));
            const newFiltered = data.new_events.filter(e => !existingIds.has(e.event_id));
            return [...newFiltered, ...prev];
          });
        }
      } catch (err) {
        console.error(err);
      }
    };

    return () => ws.close();
  }, [sessionId]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (webcamRef.current && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          wsRef.current.send(JSON.stringify({ image: imageSrc, audio_energy: 0.01 }));
        }
      }
    }, 100);

    return () => clearInterval(interval);
  }, []);

  const getRiskColor = (level) => {
    switch (level) {
      case "NORMAL": return "#10b981";
      case "LOW_RISK": return "#3b82f6";
      case "REVIEW": return "#f59e0b";
      case "HIGH_PRIORITY_REVIEW": return "#ef4444";
      case "CRITICAL_REVIEW": return "#dc2626";
      default: return "#10b981";
    }
  };

  const filteredEvents = events.filter(evt => {
    if (filterCategory === "ALL") return true;
    if (filterCategory === "FACE") return evt.event_type.includes("FACE");
    if (filterCategory === "GAZE") return evt.event_type.includes("GAZE");
    if (filterCategory === "HEAD") return evt.event_type.includes("HEAD");
    if (filterCategory === "MOUTH") return evt.event_type.includes("MOUTH");
    if (filterCategory === "OBJECTS") return evt.event_type.includes("PHONE") || evt.event_type.includes("OBJECT");
    return true;
  });

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', pb: 6 }}>
        {/* Header Bar */}
        <Paper elevation={0} sx={{ borderBottom: '1px solid rgba(255,255,255,0.1)', py: 2, px: 4, mb: 4, background: '#111827' }}>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box display="flex" alignItems="center" gap={2}>
              <Security color="primary" sx={{ fontSize: 32 }} />
              <Box>
                <Typography variant="h6" fontWeight="bold">AI Exam Proctoring Engine</Typography>
                <Typography variant="caption" color="text.secondary">ML Behavior Recognition & Bounded Evidence Pipeline</Typography>
              </Box>
            </Box>

            <Box display="flex" alignItems="center" gap={3}>
              <Chip icon={<Psychology style={{ color: '#06b6d4' }} />} label={mlPrediction.top_behavior} color="primary" variant="outlined" />
              <Chip icon={<Lightbulb style={{ color: frameQuality === 'GOOD' ? '#10b981' : '#f59e0b' }} />} label={`Quality: ${frameQuality}`} variant="outlined" color="primary" />
              <Chip icon={<Person />} label={`${studentName} (${enrollment})`} variant="outlined" color="primary" />
              <Chip
                icon={<span className="pulse-dot" style={{ backgroundColor: connected ? '#10b981' : '#ef4444' }} />}
                label={connected ? "WEBSOCKET STREAM ACTIVE" : "DISCONNECTED"}
                sx={{ background: 'rgba(255,255,255,0.05)', fontWeight: 'bold' }}
              />
            </Box>
          </Box>
        </Paper>

        {/* PRE-EXAM ALIGNMENT SETUP SCREEN */}
        {!isExamStarted ? (
          <Container maxWidth="md">
            <Paper className="glass-card" sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="h5" fontWeight="bold" gutterBottom color="primary">
                Student Face & Camera Alignment Check
              </Typography>
              <Typography variant="body2" color="text.secondary" mb={3}>
                Please align your head inside the guide oval below before starting your exam session.
              </Typography>

              <Grid container spacing={3} alignItems="center">
                <Grid item xs={12} md={7}>
                  <Box sx={{ position: 'relative', borderRadius: 3, overflow: 'hidden', border: `3px solid ${isAligned ? '#10b981' : '#ef4444'}` }}>
                    <Webcam
                      audio={false}
                      ref={webcamRef}
                      screenshotFormat="image/jpeg"
                      videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
                      style={{ width: '100%', height: 'auto', display: 'block' }}
                    />
                    
                    <svg
                      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                      viewBox="0 0 640 480"
                    >
                      <ellipse
                        cx="320"
                        cy="220"
                        rx="115"
                        ry="145"
                        fill={isAligned ? "rgba(16, 185, 129, 0.25)" : "rgba(239, 68, 68, 0.15)"}
                        stroke={isAligned ? "#10b981" : (currentStatus.distance_state === 'TOO_CLOSE' ? '#ef4444' : '#f59e0b')}
                        strokeWidth="5"
                        strokeDasharray={isAligned ? '0' : '8 4'}
                      />
                      <path
                        d="M 220 390 Q 320 310 420 390"
                        fill="none"
                        stroke={isAligned ? "#10b981" : "#f59e0b"}
                        strokeWidth="3"
                        strokeDasharray="6 3"
                      />
                    </svg>

                    <Box sx={{ position: 'absolute', bottom: 12, left: '50%', transform: 'translateX(-50%)', background: 'rgba(0,0,0,0.85)', px: 2, py: 0.8, borderRadius: 2 }}>
                      <Typography variant="caption" fontWeight="bold" color={isAligned ? '#10b981' : '#ef4444'}>
                        {isAligned
                          ? "✓ PERFECT ALIGNMENT DETECTED"
                          : (currentStatus.distance_state === 'TOO_CLOSE'
                            ? "⚠️ TOO CLOSE! STEP BACK SLIGHTLY"
                            : (currentStatus.distance_state === 'TOO_FAR'
                              ? "⚠️ TOO FAR! MOVE CLOSER"
                              : "⚠️ CENTER YOUR FACE IN OVAL"))}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>

                <Grid item xs={12} md={5}>
                  <Box display="flex" flexDirection="column" gap={2} textAlign="left">
                    <Typography variant="subtitle1" fontWeight="bold" display="flex" alignItems="center" gap={1}>
                      <Speed color="primary" /> Alignment Checklist:
                    </Typography>

                    <Alert icon={currentStatus.face_count === 1 ? <CheckCircle fontSize="inherit" /> : <Warning fontSize="inherit" />} severity={currentStatus.face_count === 1 ? "success" : "error"}>
                      Single Face: {currentStatus.face_count === 1 ? "PASSED (1 Face)" : `FAILED (${currentStatus.face_count} Faces)`}
                    </Alert>

                    <Alert icon={currentStatus.distance_state === 'OPTIMAL' ? <CheckCircle fontSize="inherit" /> : <Warning fontSize="inherit" />} severity={currentStatus.distance_state === 'OPTIMAL' ? "success" : "warning"}>
                      Face Distance: {currentStatus.distance_state} ({currentStatus.face_width_pct}% width)
                    </Alert>

                    <Alert icon={currentStatus.centering_state === 'CENTERED' ? <CheckCircle fontSize="inherit" /> : <Warning fontSize="inherit" />} severity={currentStatus.centering_state === 'CENTERED' ? "success" : "warning"}>
                      Head Centering: {currentStatus.centering_state}
                    </Alert>

                    <Alert icon={frameQuality === 'GOOD' || frameQuality === 'POOR_LIGHTING' ? <CheckCircle fontSize="inherit" /> : <Warning fontSize="inherit" />} severity={frameQuality === 'GOOD' || frameQuality === 'POOR_LIGHTING' ? "success" : "error"}>
                      Lighting & Clarity: {frameQuality}
                    </Alert>

                    <Divider sx={{ my: 1 }} />

                    <Button
                      variant="contained"
                      size="large"
                      color="success"
                      disabled={!isAligned}
                      onClick={handleStartExam}
                      startIcon={<PlayArrow />}
                      sx={{
                        py: 1.5,
                        fontWeight: 'bold',
                        fontSize: '1rem',
                        boxShadow: isAligned ? '0 0 25px rgba(16, 185, 129, 0.8)' : 'none'
                      }}
                    >
                      {isAligned ? "START EXAM SESSION" : "ALIGN FACE TO START"}
                    </Button>
                  </Box>
                </Grid>
              </Grid>
            </Paper>
          </Container>
        ) : (
          /* ACTIVE EXAM PROCTORING DASHBOARD WITH ML DEBUG TAB */
          <Container maxWidth="xl">
            {isInGracePeriod && (
              <Alert
                icon={<Timer sx={{ fontSize: 28 }} />}
                severity="info"
                sx={{ mb: 3, background: 'rgba(6, 182, 212, 0.15)', border: '1px solid #06b6d4', color: '#fff', borderRadius: 2 }}
              >
                <Typography variant="subtitle1" fontWeight="bold">
                  Exam Posture Adjustment Grace Period: {graceRemaining.toFixed(1)}s Remaining
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Please settle into your seating position. Proctoring violations will NOT be recorded during these 5 seconds.
                </Typography>
              </Alert>
            )}

            <Grid container spacing={3}>
              {/* Left Column: Live Camera & Signal Monitor */}
              <Grid item xs={12} md={5}>
                <Paper className="glass-card" sx={{ p: 2, mb: 3 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="subtitle1" fontWeight="600" display="flex" alignItems="center" gap={1}>
                      <Videocam color="primary" /> Live Student Camera Stream
                    </Typography>
                    <Chip
                      label={isInGracePeriod ? `Warmup: ${graceRemaining.toFixed(1)}s` : "Proctoring Active"}
                      size="small"
                      color={isInGracePeriod ? "warning" : "success"}
                    />
                  </Box>

                  <Box sx={{ position: 'relative', borderRadius: 2, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
                    <Webcam
                      audio={false}
                      ref={webcamRef}
                      screenshotFormat="image/jpeg"
                      videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
                      style={{ width: '100%', height: 'auto', display: 'block' }}
                    />
                  </Box>
                </Paper>

                <Paper className="glass-card" sx={{ p: 3 }}>
                  <Typography variant="subtitle1" fontWeight="600" mb={2} display="flex" alignItems="center" gap={1}>
                    <Speed color="primary" /> Live Measured Behavior Signals
                  </Typography>

                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Box p={2} borderRadius={2} sx={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                        <Typography variant="caption" color="text.secondary">Face Count</Typography>
                        <Typography variant="h6" fontWeight="bold" color={currentStatus.face_count === 1 ? '#10b981' : '#ef4444'}>
                          {currentStatus.face_count} Face ({currentStatus.face_state})
                        </Typography>
                      </Box>
                    </Grid>

                    <Grid item xs={6}>
                      <Box p={2} borderRadius={2} sx={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                        <Typography variant="caption" color="text.secondary">3D Head Pose (Yaw / Pitch)</Typography>
                        <Typography variant="h6" fontWeight="bold" color={currentStatus.head_direction === 'CENTER' ? '#10b981' : '#f59e0b'}>
                          {currentStatus.head_direction} ({currentStatus.yaw.toFixed(1)}°, {currentStatus.pitch.toFixed(1)}°)
                        </Typography>
                      </Box>
                    </Grid>

                    <Grid item xs={6}>
                      <Box p={2} borderRadius={2} sx={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                        <Typography variant="caption" color="text.secondary">Iris Gaze Ratio</Typography>
                        <Typography variant="h6" fontWeight="bold" color={currentStatus.gaze_direction === 'GAZE_CENTER' ? '#10b981' : '#f59e0b'}>
                          {currentStatus.gaze_direction} ({currentStatus.gaze_ratio.toFixed(2)})
                        </Typography>
                      </Box>
                    </Grid>

                    <Grid item xs={6}>
                      <Box p={2} borderRadius={2} sx={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                        <Typography variant="caption" color="text.secondary">Mouth Aspect Ratio (MAR)</Typography>
                        <Typography variant="h6" fontWeight="bold" color={currentStatus.mouth_state === 'NORMAL' ? '#10b981' : '#f59e0b'}>
                          MAR: {currentStatus.mar.toFixed(3)}
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>

              {/* Right Column: Evidence Card & Timeline/ML Debug Tabs */}
              <Grid item xs={12} md={7}>
                <Paper className="glass-card" sx={{ p: 3, mb: 3 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="subtitle1" fontWeight="600" display="flex" alignItems="center" gap={1}>
                      <Shield sx={{ color: getRiskColor(riskLevel) }} /> Session Evidence Score
                    </Typography>
                    <Chip
                      label={isInGracePeriod ? "WARMUP (0.0)" : riskLevel}
                      sx={{ background: getRiskColor(riskLevel), color: '#fff', fontWeight: 'bold' }}
                    />
                  </Box>

                  <Box display="flex" alignItems="baseline" gap={1} my={1}>
                    <Typography variant="h3" fontWeight="bold" sx={{ color: getRiskColor(riskLevel) }}>
                      {isInGracePeriod ? "0.0" : riskScore.toFixed(1)}
                    </Typography>
                    <Typography variant="subtitle1" color="text.secondary">/ 100.0 (Bounded Evidence Score)</Typography>
                  </Box>

                  <LinearProgress
                    variant="determinate"
                    value={isInGracePeriod ? 0 : Math.min(100, riskScore)}
                    sx={{ height: 10, borderRadius: 5, backgroundColor: 'rgba(255,255,255,0.1)', '& .MuiLinearProgress-bar': { backgroundColor: getRiskColor(riskLevel) } }}
                  />

                  {/* Single Source of Truth Category Contributions */}
                  <Box mt={2} pt={2} sx={{ borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                    <Typography variant="caption" color="text.secondary" display="flex" alignItems="center" gap={1} mb={1}>
                      <Analytics fontSize="small" /> Synchronized Category Scores & Policy Adjustments:
                    </Typography>
                    <Box display="flex" flexWrap="wrap" gap={1}>
                      <Chip size="small" label={`Objects: +${riskBreakdown.category_scores?.OBJECT || 0}`} variant="outlined" color="error" />
                      <Chip size="small" label={`Face: +${riskBreakdown.category_scores?.FACE || 0}`} variant="outlined" color="warning" />
                      <Chip size="small" label={`Gaze: +${riskBreakdown.category_scores?.GAZE || 0}`} variant="outlined" color="primary" />
                      <Chip size="small" label={`Head: +${riskBreakdown.category_scores?.HEAD || 0}`} variant="outlined" color="info" />
                      <Chip size="small" label={`Mouth: +${riskBreakdown.category_scores?.MOUTH || 0}`} variant="outlined" />
                      <Chip size="small" label={`20m Decay: -${riskBreakdown.decay_adjustment || 0}`} variant="outlined" />
                      <Chip size="small" label={`Correlation Disc: -${riskBreakdown.correlation_adjustment || 0}`} variant="outlined" />
                    </Box>
                  </Box>
                </Paper>

                <Paper className="glass-card" sx={{ p: 3 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Tabs
                      value={activeTab}
                      onChange={(e, val) => setActiveTab(val)}
                      textColor="primary"
                      indicatorColor="primary"
                    >
                      <Tab icon={<FilterList />} iconPosition="start" label={`Events (${filteredEvents.length})`} value="TIMELINE" />
                      <Tab icon={<Psychology />} iconPosition="start" label="ML Behavior Debugger" value="ML_DEBUG" />
                    </Tabs>
                  </Box>

                  {/* TAB 1: EVENTS TIMELINE */}
                  {activeTab === "TIMELINE" && (
                    <Box>
                      <Tabs
                        value={filterCategory}
                        onChange={(e, val) => setFilterCategory(val)}
                        textColor="primary"
                        indicatorColor="primary"
                        variant="scrollable"
                        sx={{ mb: 2, borderBottom: '1px solid rgba(255,255,255,0.1)' }}
                      >
                        <Tab label="ALL" value="ALL" />
                        <Tab label="FACE" value="FACE" />
                        <Tab label="GAZE" value="GAZE" />
                        <Tab label="HEAD" value="HEAD" />
                        <Tab label="MOUTH" value="MOUTH" />
                        <Tab label="OBJECTS" value="OBJECTS" />
                      </Tabs>

                      <TableContainer sx={{ maxHeight: 320 }}>
                        <Table stickyHeader size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell sx={{ background: '#141c2e' }}>EVENT</TableCell>
                              <TableCell sx={{ background: '#141c2e' }}>DURATION</TableCell>
                              <TableCell sx={{ background: '#141c2e' }}>MEASUREMENT</TableCell>
                              <TableCell sx={{ background: '#141c2e' }}>CONFIDENCE</TableCell>
                              <TableCell sx={{ background: '#141c2e' }}>EVIDENCE</TableCell>
                              <TableCell sx={{ background: '#141c2e' }}>SEVERITY</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {filteredEvents.length === 0 ? (
                              <TableRow>
                                <TableCell colSpan={6} align="center" sx={{ py: 4, color: 'text.secondary' }}>
                                  {isInGracePeriod ? "Grace period active (5.0s posture adjustment). No events logged." : "No debounced events recorded yet."}
                                </TableCell>
                              </TableRow>
                            ) : (
                              filteredEvents.map((evt, idx) => (
                                <TableRow key={evt.event_id || idx} hover onClick={() => setSelectedEvent(evt)} style={{ cursor: 'pointer' }}>
                                  <TableCell sx={{ fontWeight: 'bold', color: '#f3f4f6' }}>{evt.event_type}</TableCell>
                                  <TableCell>{(evt.duration_sec || (evt.duration_ms / 1000.0)).toFixed(1)}s</TableCell>
                                  <TableCell color="primary">{evt.measured_value || "active"}</TableCell>
                                  <TableCell>{(evt.confidence * 100).toFixed(0)}%</TableCell>
                                  <TableCell sx={{ fontWeight: 'bold', color: '#06b6d4' }}>{evt.evidence_score || 0}/100</TableCell>
                                  <TableCell>
                                    <span className={`event-chip chip-${(evt.severity || 'low').toLowerCase()}`}>
                                      {evt.severity}
                                    </span>
                                  </TableCell>
                                </TableRow>
                              ))
                            )}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Box>
                  )}

                  {/* TAB 2: ML BEHAVIOR DEBUGGER */}
                  {activeTab === "ML_DEBUG" && (
                    <Box display="flex" flexDirection="column" gap={2}>
                      <Box p={2} borderRadius={2} sx={{ background: 'rgba(6, 182, 212, 0.1)', border: '1px solid #06b6d4' }}>
                        <Typography variant="subtitle2" color="primary" fontWeight="bold" display="flex" alignItems="center" gap={1}>
                          <AutoFixHigh fontSize="small" /> Active ML Behavior Model:
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {mlPrediction.model_version || "XGBoost Temporal Classifier v1.1.0"}
                        </Typography>
                      </Box>

                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="caption" color="text.secondary" mb={1} display="block">
                            Class Probabilities (5.0s Sliding Window):
                          </Typography>
                          {Object.entries(mlPrediction.probabilities || {}).map(([cls, prob]) => (
                            <Box key={cls} mb={1}>
                              <Box display="flex" justifyContent="space-between" mb={0.5}>
                                <Typography variant="caption" fontWeight="600">{cls}</Typography>
                                <Typography variant="caption" color="primary">{(prob * 100).toFixed(0)}%</Typography>
                              </Box>
                              <LinearProgress variant="determinate" value={prob * 100} sx={{ height: 6, borderRadius: 3 }} />
                            </Box>
                          ))}
                        </Grid>

                        <Grid item xs={6}>
                          <Typography variant="caption" color="text.secondary" mb={1} display="block">
                            Extracted 25 Temporal Features (Summary):
                          </Typography>
                          <Box p={2} borderRadius={2} sx={{ background: 'rgba(0,0,0,0.3)', maxHeight: 220, overflowY: 'auto' }}>
                            {Object.entries(temporalFeatures || {}).map(([k, v]) => (
                              <Box key={k} display="flex" justifyContent="space-between" py={0.3} sx={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                <Typography variant="caption" color="text.secondary">{k}:</Typography>
                                <Typography variant="caption" fontWeight="bold">{typeof v === 'number' ? v.toFixed(2) : v}</Typography>
                              </Box>
                            ))}
                          </Box>
                        </Grid>
                      </Grid>
                    </Box>
                  )}
                </Paper>
              </Grid>
            </Grid>
          </Container>
        )}

        {/* 14-Field Detailed Inspection Modal */}
        <Dialog open={Boolean(selectedEvent)} onClose={() => setSelectedEvent(null)} maxWidth="sm" fullWidth>
          <DialogTitle display="flex" alignItems="center" gap={1}>
            <Info color="primary" /> Event Inspection Detail
          </DialogTitle>
          <DialogContent dividers>
            {selectedEvent && (
              <Box display="flex" flexDirection="column" gap={1.2}>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">1. Event Type:</Typography>
                  <Typography fontWeight="bold">{selectedEvent.event_type}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">2. Start Time:</Typography>
                  <Typography fontWeight="bold">{selectedEvent.started_at || "N/A"}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">3. End Time:</Typography>
                  <Typography fontWeight="bold">{selectedEvent.ended_at || "N/A"}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">4. Duration:</Typography>
                  <Typography fontWeight="bold">{(selectedEvent.duration_sec || (selectedEvent.duration_ms / 1000.0)).toFixed(1)} seconds</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">5. Measured Value:</Typography>
                  <Typography fontWeight="bold" color="primary">{selectedEvent.measured_value || "N/A"}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">6. Raw Detector Confidence:</Typography>
                  <Typography fontWeight="bold">{(selectedEvent.confidence * 100).toFixed(0)}%</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">7. Operating Threshold:</Typography>
                  <Typography fontWeight="bold">0.55</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">8. Temporal Persistence:</Typography>
                  <Typography fontWeight="bold">{((selectedEvent.temporal_persistence || 1.0) * 100).toFixed(0)}%</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">9. Frame Quality State:</Typography>
                  <Typography fontWeight="bold" color="primary">{frameQuality}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">10. Event Evidence Score:</Typography>
                  <Typography fontWeight="bold" color="primary">{selectedEvent.evidence_score || 0} / 100</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">11. Policy Severity (Cap):</Typography>
                  <Chip size="small" label={`${selectedEvent.severity} (Cap: ${selectedEvent.severity_cap || 25})`} color="primary" />
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">12. Detector Name:</Typography>
                  <Typography fontWeight="bold">MediaPipe FaceMesh / YOLOv8</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">13. Model Version:</Typography>
                  <Typography fontWeight="bold">{selectedEvent.model_version || "MediaPipe 0.10.14"}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography color="text.secondary">14. Threshold Config Version:</Typography>
                  <Typography fontWeight="bold">{selectedEvent.threshold_config_version || "v1.2.0"}</Typography>
                </Box>
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSelectedEvent(null)}>Close</Button>
          </DialogActions>
        </Dialog>
      </Box>
    </ThemeProvider>
  );
}
