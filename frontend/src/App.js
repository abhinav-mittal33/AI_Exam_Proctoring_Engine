import React, { useState, useEffect, useRef } from 'react';
import Webcam from 'react-webcam';
import {
  ThemeProvider, createTheme, CssBaseline, Container, Grid, Paper, Typography, Box,
  Chip, LinearProgress, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Tabs, Tab
} from '@mui/material';
import {
  Videocam, Security, Shield, Speed, FilterList, Person
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

  const [connected, setConnected] = useState(false);
  const [riskScore, setRiskScore] = useState(0.0);
  const [riskLevel, setRiskLevel] = useState("NORMAL");
  
  const [currentStatus, setCurrentStatus] = useState({
    face_count: 1,
    face_state: "ONE_FACE",
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

  const webcamRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    fetch(`${API_HOST}/api/proctor/session/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, student_id: enrollment, exam_id: "EXAM_2026", student_name: studentName })
    }).catch(console.error);

    const ws = new WebSocket(`${WS_HOST}/ws/proctor/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data);
        if (data.current_status) setCurrentStatus(data.current_status);
        if (data.risk_score !== undefined) setRiskScore(data.risk_score);
        if (data.risk_level) setRiskLevel(data.risk_level);

        if (data.new_events && data.new_events.length > 0) {
          setEvents(prev => [...data.new_events, ...prev]);
        }
      } catch (err) {
        console.error(err);
      }
    };

    return () => ws.close();
  }, [sessionId, studentName, enrollment]);

  // Capture video frame at 10 FPS
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
        <Paper elevation={0} sx={{ borderBottom: '1px solid rgba(255,255,255,0.1)', py: 2, px: 4, mb: 4, background: '#111827' }}>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box display="flex" alignItems="center" gap={2}>
              <Security color="primary" sx={{ fontSize: 32 }} />
              <Box>
                <Typography variant="h6" fontWeight="bold">AI Exam Proctoring Engine</Typography>
                <Typography variant="caption" color="text.secondary">MediaPipe 468 3D Landmarks & Ultralytics YOLOv8 Detection</Typography>
              </Box>
            </Box>

            <Box display="flex" alignItems="center" gap={3}>
              <Chip icon={<Person />} label={`${studentName} (${enrollment})`} variant="outlined" color="primary" />
              <Chip
                icon={<span className="pulse-dot" style={{ backgroundColor: connected ? '#10b981' : '#ef4444' }} />}
                label={connected ? "WEBSOCKET STREAM ACTIVE" : "DISCONNECTED"}
                sx={{ background: 'rgba(255,255,255,0.05)', fontWeight: 'bold' }}
              />
            </Box>
          </Box>
        </Paper>

        <Container maxWidth="xl">
          <Grid container spacing={3}>
            {/* Left Column: Live Camera Feed & Real-time Computer Vision Signals */}
            <Grid item xs={12} md={6}>
              <Paper className="glass-card" sx={{ p: 2, mb: 3 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="subtitle1" fontWeight="600" display="flex" alignItems="center" gap={1}>
                    <Videocam color="primary" /> Live Student Camera Feed
                  </Typography>
                  <Chip label="MediaPipe + YOLOv8 Active" size="small" color="secondary" />
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

              {/* Real-time Observable Computer Vision Signals */}
              <Paper className="glass-card" sx={{ p: 3 }}>
                <Typography variant="subtitle1" fontWeight="600" mb={2} display="flex" alignItems="center" gap={1}>
                  <Speed color="primary" /> MediaPipe & YOLOv8 Computer Vision Signals
                </Typography>

                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Box p={2} borderRadius={2} sx={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                      <Typography variant="caption" color="text.secondary">Face Presence Count</Typography>
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
                      <Typography variant="caption" color="text.secondary">Iris Gaze Displacement</Typography>
                      <Typography variant="h6" fontWeight="bold" color={currentStatus.gaze_direction === 'GAZE_CENTER' ? '#10b981' : '#f59e0b'}>
                        {currentStatus.gaze_direction} (Ratio: {currentStatus.gaze_ratio.toFixed(2)})
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={6}>
                    <Box p={2} borderRadius={2} sx={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                      <Typography variant="caption" color="text.secondary">Mouth Aspect Ratio (MAR)</Typography>
                      <Typography variant="h6" fontWeight="bold" color={currentStatus.mouth_state === 'NORMAL' ? '#10b981' : '#f59e0b'}>
                        MAR: {currentStatus.mar.toFixed(3)} ({currentStatus.mouth_state})
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12}>
                    <Box p={2} borderRadius={2} sx={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                      <Typography variant="caption" color="text.secondary">YOLOv8 Prohibited Object Detection</Typography>
                      {currentStatus.prohibited_objects.length === 0 ? (
                        <Typography variant="h6" fontWeight="bold" color="#10b981">
                          CLEAR — No prohibited objects detected
                        </Typography>
                      ) : (
                        currentStatus.prohibited_objects.map((obj, idx) => (
                          <Typography key={idx} variant="h6" fontWeight="bold" color="#ef4444">
                            🚨 {obj.object_name.toUpperCase()} DETECTED (Confidence: {(obj.confidence * 100).toFixed(0)}%)
                          </Typography>
                        ))
                      )}
                    </Box>
                  </Grid>
                </Grid>
              </Paper>
            </Grid>

            {/* Right Column: Evidence Risk Gauge & Observable Event Log */}
            <Grid item xs={12} md={6}>
              <Paper className="glass-card" sx={{ p: 3, mb: 3 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="subtitle1" fontWeight="600" display="flex" alignItems="center" gap={1}>
                    <Shield sx={{ color: getRiskColor(riskLevel) }} /> Evidence Risk Score
                  </Typography>
                  <Chip
                    label={riskLevel}
                    sx={{ background: getRiskColor(riskLevel), color: '#fff', fontWeight: 'bold' }}
                  />
                </Box>

                <Box display="flex" alignItems="baseline" gap={1} my={1}>
                  <Typography variant="h3" fontWeight="bold" sx={{ color: getRiskColor(riskLevel) }}>
                    {riskScore}
                  </Typography>
                  <Typography variant="subtitle1" color="text.secondary">/ 100.0</Typography>
                </Box>

                <LinearProgress
                  variant="determinate"
                  value={Math.min(100, riskScore)}
                  sx={{ height: 10, borderRadius: 5, backgroundColor: 'rgba(255,255,255,0.1)', '& .MuiLinearProgress-bar': { backgroundColor: getRiskColor(riskLevel) } }}
                />
              </Paper>

              <Paper className="glass-card" sx={{ p: 3 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="subtitle1" fontWeight="600" display="flex" alignItems="center" gap={1}>
                    <FilterList color="primary" /> Debounced Proctoring Events ({filteredEvents.length})
                  </Typography>
                </Box>

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

                <TableContainer sx={{ maxHeight: 380 }}>
                  <Table stickyHeader size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ background: '#141c2e' }}>Event</TableCell>
                        <TableCell sx={{ background: '#141c2e' }}>Duration</TableCell>
                        <TableCell sx={{ background: '#141c2e' }}>Confidence</TableCell>
                        <TableCell sx={{ background: '#141c2e' }}>Severity</TableCell>
                        <TableCell sx={{ background: '#141c2e' }}>Time</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {filteredEvents.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={5} align="center" sx={{ py: 4, color: 'text.secondary' }}>
                            No debounced events recorded yet.
                          </TableCell>
                        </TableRow>
                      ) : (
                        filteredEvents.map((evt, idx) => (
                          <TableRow key={evt.event_id || idx} hover>
                            <TableCell sx={{ fontWeight: 'bold', color: '#f3f4f6' }}>{evt.event_type}</TableCell>
                            <TableCell>{(evt.duration_ms / 1000.0).toFixed(1)}s</TableCell>
                            <TableCell>{(evt.confidence * 100).toFixed(0)}%</TableCell>
                            <TableCell>
                              <span className={`event-chip chip-${(evt.severity || 'low').toLowerCase()}`}>
                                {evt.severity}
                              </span>
                            </TableCell>
                            <TableCell sx={{ color: 'text.secondary', fontSize: '0.75rem' }}>{evt.ended_at}</TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  );
}
