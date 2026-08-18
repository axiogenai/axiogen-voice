"""
metrics.py - Performance Instrumentation & Metrics Collector for Axiogen Voice TTS.

Tracks:
- Time-to-First-Byte (TTFB) / Time-to-First-Audio (TTFA)
- Per-sentence generation latency
- Total generation & streaming duration
- Queue wait time & throughput
- Prometheus & JSON export
"""

import time
import threading
from typing import Dict, Any, List
from collections import deque

class MetricsCollector:
    def __init__(self, max_history: int = 1000):
        self._lock = threading.Lock()
        self.total_requests = 0
        self.active_streams = 0
        self.total_sentences_generated = 0
        self.total_audio_seconds = 0.0
        
        # Histograms / recent samples
        self.ttfb_history = deque(maxlen=max_history)
        self.sentence_latencies = deque(maxlen=max_history)
        self.total_duration_history = deque(maxlen=max_history)

    def record_request_start(self):
        with self._lock:
            self.total_requests += 1
            self.active_streams += 1

    def record_request_end(self):
        with self._lock:
            if self.active_streams > 0:
                self.active_streams -= 1

    def record_ttfb(self, ttfb_ms: float):
        with self._lock:
            self.ttfb_history.append(ttfb_ms)

    def record_sentence(self, latency_ms: float, audio_duration_sec: float):
        with self._lock:
            self.total_sentences_generated += 1
            self.sentence_latencies.append(latency_ms)
            self.total_audio_seconds += audio_duration_sec

    def record_total_duration(self, duration_ms: float):
        with self._lock:
            self.total_duration_history.append(duration_ms)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            avg_ttfb = sum(self.ttfb_history) / len(self.ttfb_history) if self.ttfb_history else 0.0
            p95_ttfb = sorted(self.ttfb_history)[int(len(self.ttfb_history) * 0.95)] if self.ttfb_history else 0.0
            avg_sentence_gen = sum(self.sentence_latencies) / len(self.sentence_latencies) if self.sentence_latencies else 0.0
            avg_total_dur = sum(self.total_duration_history) / len(self.total_duration_history) if self.total_duration_history else 0.0

            return {
                "status": "healthy",
                "active_streams": self.active_streams,
                "total_requests": self.total_requests,
                "total_sentences_generated": self.total_sentences_generated,
                "total_audio_seconds_generated": round(self.total_audio_seconds, 2),
                "latency_metrics": {
                    "avg_ttfb_ms": round(avg_ttfb, 2),
                    "p95_ttfb_ms": round(p95_ttfb, 2),
                    "avg_sentence_gen_ms": round(avg_sentence_gen, 2),
                    "avg_total_stream_ms": round(avg_total_dur, 2)
                }
            }

    def get_prometheus_metrics(self) -> str:
        summary = self.get_summary()
        m = summary["latency_metrics"]
        lines = [
            "# HELP axiogen_tts_active_streams Current active streaming sessions",
            "# TYPE axiogen_tts_active_streams gauge",
            f"axiogen_tts_active_streams {summary['active_streams']}",
            "# HELP axiogen_tts_requests_total Total TTS requests received",
            "# TYPE axiogen_tts_requests_total counter",
            f"axiogen_tts_requests_total {summary['total_requests']}",
            "# HELP axiogen_tts_sentences_total Total sentences synthesized",
            "# TYPE axiogen_tts_sentences_total counter",
            f"axiogen_tts_sentences_total {summary['total_sentences_generated']}",
            "# HELP axiogen_tts_audio_seconds_total Total seconds of audio generated",
            "# TYPE axiogen_tts_audio_seconds_total counter",
            f"axiogen_tts_audio_seconds_total {summary['total_audio_seconds_generated']}",
            "# HELP axiogen_tts_ttfb_ms_avg Average time to first audio byte in ms",
            "# TYPE axiogen_tts_ttfb_ms_avg gauge",
            f"axiogen_tts_ttfb_ms_avg {m['avg_ttfb_ms']}",
            "# HELP axiogen_tts_sentence_gen_ms_avg Average sentence synthesis duration in ms",
            "# TYPE axiogen_tts_sentence_gen_ms_avg gauge",
            f"axiogen_tts_sentence_gen_ms_avg {m['avg_sentence_gen_ms']}"
        ]
        return "\n".join(lines) + "\n"

# Global metrics instance
GLOBAL_METRICS = MetricsCollector()
