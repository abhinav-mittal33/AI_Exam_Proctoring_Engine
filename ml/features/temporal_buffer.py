import time
from typing import Dict, List

class TemporalBuffer:
    def __init__(self, window_duration_sec: float = 5.0):
        self.window_duration_sec = window_duration_sec
        # Session ID -> list of canonical timestep dicts
        self.buffers: Dict[str, List[dict]] = {}

    def add_timestep(self, session_id: str, timestep_dict: dict):
        if session_id not in self.buffers:
            self.buffers[session_id] = []

        buf = self.buffers[session_id]
        buf.append(timestep_dict)
        now = time.time()

        # Evict timesteps older than window_duration_sec
        self.buffers[session_id] = [ts for ts in buf if (now - ts.get("timestamp", now)) <= self.window_duration_sec]

    def get_window_timesteps(self, session_id: str) -> List[dict]:
        return self.buffers.get(session_id, [])

    def clear(self, session_id: str):
        if session_id in self.buffers:
            del self.buffers[session_id]

temporal_buffer = TemporalBuffer()
