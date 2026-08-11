import numpy as np

class AudioDetector:
    def __init__(self):
        pass

    def analyze_audio(self, pcm_data: bytes = None, energy: float = 0.0) -> dict:
        """
        Analyzes audio signal for voice activity & background noise level.
        """
        if pcm_data:
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            if len(audio_array) > 0:
                energy = float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)) / 32768.0)

        is_voice = energy > 0.05
        state = "AUDIO_ACTIVITY" if is_voice else "SILENCE"

        return {
            "energy": round(float(energy), 4),
            "is_voice": is_voice,
            "state": state,
            "confidence": 0.85 if is_voice else 0.95
        }

audio_detector = AudioDetector()
