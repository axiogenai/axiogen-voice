"""
server.py - Production Low-Latency Streaming TTS Server for Axiogen Voice (Kokoro 82M).

Architecture:
- Global `pipeline = KPipeline(...)` loaded ONCE at startup (never reloaded per request).
- `@app.post("/tts")` and `@app.post("/v1/audio/speech")` for streaming and batch synthesis.
- `@app.websocket("/ws/tts")` for real-time WebSocket sentence streaming.
- Dynamic `/v1/voices`, `/health`, and `/metrics`.
"""

import os
import io
import json
import time
import secrets
import struct
import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Header, Request, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, FileResponse, HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import soundfile as sf
import numpy as np

from config import CONFIG
from splitter import split_sentences, IncrementalSentenceBuffer
from metrics import GLOBAL_METRICS

# Try PyTorch KPipeline or fallback to optimized ONNX engine
try:
    from kokoro import KPipeline
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Global pipelines loaded ONCE at module level
    pipelines = {
        'a': KPipeline(lang_code='a', device=device),
        'b': KPipeline(lang_code='b', device=device)
    }
    pipeline = pipelines['a']
    USE_PYTORCH_PIPELINE = True
    print(f"[INIT] Kokoro PyTorch KPipeline loaded on {device}.")
except Exception as e:
    from streaming import KokoroStreamingEngine
    USE_PYTORCH_PIPELINE = False
    engine = KokoroStreamingEngine(CONFIG)
    pipeline = engine
    print(f"[INIT] Kokoro ONNX Streaming Engine loaded: {e}")

START_TIME = time.time()

def load_api_keys():
    if os.path.exists(CONFIG.api_keys_file):
        try:
            with open(CONFIG.api_keys_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_api_keys(keys):
    with open(CONFIG.api_keys_file, "w") as f:
        json.dump(keys, f, indent=4)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform warm-up inference at startup to prime caches
    print("[WARMUP] Warming up neural model caches...")
    t0 = time.perf_counter()
    try:
        if USE_PYTORCH_PIPELINE:
            for _, ps, _ in pipeline("System warmup.", voice=CONFIG.default_voice, speed=1.0):
                pass
        else:
            engine.warm_up()
        print(f"[WARMUP] Completed in {(time.perf_counter() - t0)*1000:.2f}ms. Ready for requests.")
    except Exception as ex:
        print(f"[WARMUP] Notice: {ex}")
    
    yield
    print("[SHUTDOWN] Server shutting down...")

app = FastAPI(
    title="Axiogen Voice Streaming API",
    description="Low-Latency Streaming Text-to-Speech Engine",
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def verify_api_key(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Query(None)
):
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1].strip()
    elif authorization:
        token = authorization.strip()
    elif x_api_key:
        token = x_api_key.strip()
    elif api_key:
        token = api_key.strip()

    referer = request.headers.get("referer", "")
    origin = request.headers.get("origin", "")
    host = request.headers.get("host", "")
    if not token and (host in referer or host in origin or "vercel.app" in referer or "axiogen" in referer):
        return CONFIG.admin_key

    if not token or token == CONFIG.admin_key:
        return CONFIG.admin_key

    keys = load_api_keys()
    if token in keys:
        return token

    raise HTTPException(status_code=401, detail={"error": {"message": "Invalid API Key"}})

class SpeechRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Input text to synthesize")
    input: Optional[str] = Field(default=None, description="OpenAI-compatible text field")
    voice: Optional[str] = Field(default=None)
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
    stream: bool = False
    format: str = Field(default="wav", description="'wav' or 'pcm'")

def synthesize_sentence_sync(text: str, voice: str, speed: float):
    """Synthesizes a single sentence using the persistent pipeline without reloading."""
    v = voice or CONFIG.default_voice
    sp = max(0.5, min(float(speed or 1.0), 2.0))

    if USE_PYTORCH_PIPELINE:
        lang_code = v[0] if (v and v[0] in 'ab') else 'a'
        p = pipelines.get(lang_code, pipeline)
        all_samples = []
        for _, _, audio in p(text, voice=v, speed=sp):
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            elif hasattr(audio, 'numpy'):
                audio = audio.numpy()
            all_samples.append(audio)
        
        samples = np.concatenate(all_samples) if all_samples else np.zeros(2400, dtype=np.float32)
        sr = 24000
    else:
        samples, sr = engine.engine.create(text, voice=v, speed=sp)

    pcm_data = (samples * 32767.0).clip(-32768, 32767).astype(np.int16)
    pcm_bytes = pcm_data.tobytes()

    wav_buf = io.BytesIO()
    sf.write(wav_buf, samples, sr, format='WAV', subtype='PCM_16')
    wav_bytes = wav_buf.getvalue()
    dur_ms = (len(samples) / sr) * 1000.0

    return pcm_bytes, wav_bytes, dur_ms, sr

async def stream_audio_generator(text: str, voice: str, speed: float) -> AsyncGenerator[bytes, None]:
    """HTTP Chunked streaming generator yielding framed WAV chunks per sentence."""
    sentences = split_sentences(text)
    for s in sentences:
        _, wav_b, _, _ = await asyncio.to_thread(synthesize_sentence_sync, s, voice, speed)
        yield struct.pack(">I", len(wav_b)) + wav_b

# ── POST /tts (Primary Sentence Streaming Endpoint) ──
@app.post("/tts")
@app.post("/v1/audio/speech")
async def tts(request: SpeechRequest, _: str = Depends(verify_api_key)):
    """
    Main TTS endpoint.
    Uses persistent `pipeline = KPipeline(...)`.
    Streams sentence-by-sentence if stream=True, or returns combined audio if stream=False.
    """
    raw_text = request.text or request.input or ""
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    voice = request.voice or CONFIG.default_voice
    speed = float(request.speed or 1.0)

    # If stream requested, stream sentence chunks progressively
    if request.stream:
        return StreamingResponse(
            stream_audio_generator(raw_text, voice, speed),
            media_type="application/octet-stream"
        )

    # Non-streaming: synthesize sentences and return single WAV
    sentences = split_sentences(raw_text)
    all_pcm = []
    sr = 24000
    for s in sentences:
        pcm_b, _, _, sample_rate = await asyncio.to_thread(synthesize_sentence_sync, s, voice, speed)
        all_pcm.append(pcm_b)
        sr = sample_rate

    raw_bytes = b"".join(all_pcm)
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format='WAV', subtype='PCM_16')
    return Response(content=buf.getvalue(), media_type="audio/wav")

# ── WEBSOCKET REAL-TIME STREAMING ENDPOINT ──
@app.websocket("/ws/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = f"ws_{int(time.time()*1000)}"
    buffer = IncrementalSentenceBuffer()

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type", "generate")
            voice = data.get("voice", CONFIG.default_voice)
            speed = float(data.get("speed", 1.0))
            out_format = data.get("format", "wav").lower()

            if msg_type == "generate" or "text" in data:
                text = (data.get("text") or "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "message": "Empty text"})
                    continue

                await websocket.send_json({"type": "start", "session_id": session_id})
                sentences = split_sentences(text)
                
                for idx, sent in enumerate(sentences):
                    t0 = time.perf_counter()
                    pcm_b, wav_b, dur_ms, sr = await asyncio.to_thread(synthesize_sentence_sync, sent, voice, speed)
                    gen_ms = (time.perf_counter() - t0) * 1000
                    payload = wav_b if out_format == "wav" else pcm_b

                    await websocket.send_json({
                        "type": "audio_meta",
                        "sentence_index": idx,
                        "text": sent,
                        "duration_ms": round(dur_ms, 1),
                        "gen_time_ms": round(gen_ms, 1),
                        "is_final": (idx == len(sentences) - 1)
                    })
                    await websocket.send_bytes(payload)

                await websocket.send_json({"type": "done", "total_sentences": len(sentences)})

            elif msg_type == "token":
                token = data.get("data", "")
                for sent in buffer.add_token(token):
                    pcm_b, wav_b, dur_ms, sr = await asyncio.to_thread(synthesize_sentence_sync, sent, voice, speed)
                    payload = wav_b if out_format == "wav" else pcm_b
                    await websocket.send_json({"type": "audio_meta", "text": sent, "duration_ms": round(dur_ms, 1)})
                    await websocket.send_bytes(payload)

            elif msg_type == "flush":
                rem = buffer.flush()
                if rem:
                    pcm_b, wav_b, dur_ms, sr = await asyncio.to_thread(synthesize_sentence_sync, rem, voice, speed)
                    payload = wav_b if out_format == "wav" else pcm_b
                    await websocket.send_json({"type": "audio_meta", "text": rem, "duration_ms": round(dur_ms, 1), "is_final": True})
                    await websocket.send_bytes(payload)
                await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[{session_id}] WebSocket error: {e}")

# ── HEALTH, VOICES & METRICS ──

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "engine": "axiogen-v2-streaming",
        "pipeline_loaded": True,
        "uptime_seconds": int(time.time() - START_TIME)
    }

@app.get("/metrics")
async def metrics(format: str = Query("json")):
    if format.lower() == "prometheus":
        return PlainTextResponse(GLOBAL_METRICS.get_prometheus_metrics())
    return GLOBAL_METRICS.get_summary()

@app.get("/v1/voices")
async def list_voices(_: str = Depends(verify_api_key)):
    voice_list = ["af_bella", "af_heart", "af_sarah", "af_nicole", "af_sky", "am_adam", "am_michael", "bf_emma", "bm_george"]
    voices_data = []
    for v in voice_list:
        parts = v.split("_")
        gender = "Female" if parts[0].endswith("f") else "Male"
        accent = "American" if parts[0].startswith("a") else "British"
        name = parts[1].capitalize() if len(parts) > 1 else v
        voices_data.append({"voice_id": v, "name": name, "accent": accent, "gender": gender, "style": "Neural"})
    return {"voices": voices_data}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h1>Axiogen Voice Streaming API</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=CONFIG.host, port=CONFIG.port)
