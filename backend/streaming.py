"""
streaming.py - Low-Latency Audio Streaming Pipeline & Producer-Consumer Queue.

Architecture:
User Text -> Sentence Splitter -> Sentence Queue -> Kokoro Worker (Producer) -> AudioChunkQueue -> WebSocket/HTTP Streamer (Consumer)
"""

import os
import io
import time
import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional, Tuple, Dict, Any

import soundfile as sf
import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

from config import CONFIG
from splitter import split_sentences
from metrics import GLOBAL_METRICS

logger = logging.getLogger("axiogen.tts")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

@dataclass
class AudioChunk:
    sentence_idx: int
    text: str
    pcm_bytes: bytes       # Raw 16-bit PCM (24000Hz, 1 channel)
    wav_bytes: bytes       # Self-contained RIFF/WAV binary
    sample_rate: int
    duration_ms: float
    gen_time_ms: float
    is_final: bool

class KokoroStreamingEngine:
    def __init__(self, config=CONFIG):
        self.config = config
        self.engine: Optional[Kokoro] = None
        self.sample_rate = config.sample_rate
        self.available_voices: List[str] = []

    def initialize(self):
        """Initializes and optimizes the ONNX runtime session once."""
        if self.engine is not None:
            return

        logger.info(f"[INIT] Loading Kokoro ONNX model from {self.config.model_path}...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.config.num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        try:
            session = ort.InferenceSession(
                self.config.model_path,
                sess_options=sess_options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.engine = Kokoro.from_session(session, self.config.voices_path)
            logger.info("[INIT] Initialized Kokoro from optimized ONNX session.")
        except Exception as e:
            logger.warning(f"[INIT] Fallback to standard Kokoro loader: {e}")
            self.engine = Kokoro(self.config.model_path, self.config.voices_path)

        try:
            self.available_voices = sorted(list(self.engine.get_voices()))
        except Exception:
            self.available_voices = [self.config.default_voice]

        logger.info(f"[INIT] Kokoro Engine loaded successfully with {len(self.available_voices)} voices.")

    def warm_up(self):
        """Pre-heats model caches & execution providers at startup to eliminate first-request lag."""
        if not self.config.enable_warmup:
            return

        if not self.engine:
            self.initialize()
        
        logger.info("[WARMUP] Running neural model warm-up...")
        t0 = time.perf_counter()
        try:
            self.engine.create(
                "System initialized. Welcome to Axiogen Voice.", 
                voice=self.config.default_voice, 
                speed=self.config.default_speed, 
                lang="en-us"
            )
            dur = (time.perf_counter() - t0) * 1000
            logger.info(f"[WARMUP] Model warm-up complete in {dur:.2f}ms. Caches primed.")
        except Exception as e:
            logger.error(f"[WARMUP] Warm-up warning: {e}")

    def synthesize_sentence_sync(self, text: str, voice: Optional[str], speed: Optional[float]) -> Tuple[bytes, bytes, float]:
        """Synchronous synthesis of a single sentence -> (pcm_bytes, wav_bytes, duration_ms)."""
        v = voice if (voice and voice in self.available_voices) else self.config.default_voice
        lang = "en-gb" if v.startswith("b") else "en-us"
        sp = max(0.5, min(float(speed or self.config.default_speed), 2.0))

        samples, sr = self.engine.create(text, voice=v, speed=sp, lang=lang)
        
        # Ensure 16-bit PCM integer scaling
        pcm_data = (samples * 32767.0).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = pcm_data.tobytes()

        # Build WAV container
        wav_buf = io.BytesIO()
        sf.write(wav_buf, samples, sr, format='WAV', subtype='PCM_16')
        wav_bytes = wav_buf.getvalue()

        duration_ms = (len(samples) / sr) * 1000.0
        return pcm_bytes, wav_bytes, duration_ms

    async def stream_pipeline(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        request_id: Optional[str] = None
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        True Asynchronous Producer-Consumer Pipeline:
        - Producer: Kokoro Worker Task generates audio sentence-by-sentence into AudioQueue.
        - Consumer: Yields AudioChunks immediately as they arrive.
        """
        req_id = request_id or f"req_{int(time.time()*1000)}"
        t_req_start = time.perf_counter()
        GLOBAL_METRICS.record_request_start()

        # Step 1: Sentence Splitting
        t_split_start = time.perf_counter()
        sentences = split_sentences(text, max_words_per_chunk=self.config.max_sentence_words)
        t_split_end = time.perf_counter()
        split_dur_ms = (t_split_end - t_split_start) * 1000
        
        logger.info(f"[{req_id}] Text received ({len(text)} chars) -> Split into {len(sentences)} sentences in {split_dur_ms:.2f}ms")

        if not sentences:
            GLOBAL_METRICS.record_request_end()
            return

        # Step 2: Thread-Safe Async Audio Queue (Capacity configurable for backpressure)
        audio_queue: asyncio.Queue[Optional[AudioChunk]] = asyncio.Queue(maxsize=self.config.queue_maxsize)

        # Step 3: Producer Task (Kokoro Worker)
        async def producer_worker():
            for idx, sentence in enumerate(sentences):
                t_gen_start = time.perf_counter()
                
                # Run ONNX inference in non-blocking threadpool
                pcm_b, wav_b, dur_ms = await asyncio.to_thread(
                    self.synthesize_sentence_sync, sentence, voice, speed
                )
                
                t_gen_end = time.perf_counter()
                gen_dur_ms = (t_gen_end - t_gen_start) * 1000
                is_final = (idx == len(sentences) - 1)

                chunk = AudioChunk(
                    sentence_idx=idx,
                    text=sentence,
                    pcm_bytes=pcm_b,
                    wav_bytes=wav_b,
                    sample_rate=self.sample_rate,
                    duration_ms=dur_ms,
                    gen_time_ms=gen_dur_ms,
                    is_final=is_final
                )

                logger.info(f"[{req_id}] Sentence {idx+1}/{len(sentences)} generated in {gen_dur_ms:.1f}ms: \"{sentence[:30]}...\" (Audio: {dur_ms:.0f}ms)")
                GLOBAL_METRICS.record_sentence(gen_dur_ms, dur_ms / 1000.0)

                # Push to consumer queue (respects backpressure)
                await audio_queue.put(chunk)

            # Signal EOF
            await audio_queue.put(None)

        # Launch producer in background
        producer_task = asyncio.create_task(producer_worker())

        # Step 4: Consumer (Stream Chunks Immediately)
        first_chunk = True
        try:
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break

                if first_chunk:
                    ttfb_ms = (time.perf_counter() - t_req_start) * 1000
                    logger.info(f"[{req_id}] ⚡ [FIRST-AUDIO-SENT] TTFB: {ttfb_ms:.1f}ms (Client starts playback now)")
                    GLOBAL_METRICS.record_ttfb(ttfb_ms)
                    first_chunk = False

                yield chunk
                audio_queue.task_done()

            await producer_task

        finally:
            total_time_ms = (time.perf_counter() - t_req_start) * 1000
            GLOBAL_METRICS.record_total_duration(total_time_ms)
            GLOBAL_METRICS.record_request_end()
            logger.info(f"[{req_id}] Stream completed in {total_time_ms:.1f}ms")
