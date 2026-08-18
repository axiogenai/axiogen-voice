"""
server.py - Production-Ready Low-Latency Streaming TTS Server for Axiogen Voice (Kokoro 82M).

Endpoints:
- WS   /ws/tts            -> Real-time sentence-by-sentence WebSocket streaming
- POST /v1/audio/speech   -> OpenAI-compatible TTS (supports stream: true / false)
- GET  /v1/voices         -> Dynamically discovered voice models with metadata
- GET  /v1/models         -> Available model list
- GET  /metrics           -> Live Prometheus and JSON performance metrics
- GET  /health            -> System health & uptime
- GET  /                  -> Real-time Voice Studio Playground
"""

import os
import json
import time
import secrets
import struct
import urllib.request
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Header, Request, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, FileResponse, HTMLResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import CONFIG
from splitter import IncrementalSentenceBuffer
from metrics import GLOBAL_METRICS
from streaming import KokoroStreamingEngine, AudioChunk

# Global Streaming Engine (loaded once in lifespan)
engine = KokoroStreamingEngine(CONFIG)
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

def ensure_model_files():
    os.makedirs(CONFIG.models_dir, exist_ok=True)
    if not os.path.exists(CONFIG.model_path):
        print(f"[DOWNLOAD] Downloading model from {CONFIG.model_url} -> {CONFIG.model_path}...")
        urllib.request.urlretrieve(CONFIG.model_url, CONFIG.model_path)
    if not os.path.exists(CONFIG.voices_path):
        print(f"[DOWNLOAD] Downloading voices from {CONFIG.voices_url} -> {CONFIG.voices_path}...")
        urllib.request.urlretrieve(CONFIG.voices_url, CONFIG.voices_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Ensure model files exist on disk
    ensure_model_files()
    
    # 2. Initialize Kokoro engine (Persistent singleton)
    engine.initialize()
    
    # 3. Model warm-up
    engine.warm_up()
    
    print(f"[READY] Axiogen Voice Streaming Server initialized on {CONFIG.host}:{CONFIG.port}")
    yield
    print("[SHUTDOWN] Server shutting down...")

app = FastAPI(
    title="Axiogen Voice Streaming API",
    description="Low-Latency Streaming Neural Text-to-Speech Engine",
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

    # Allow browser playground requests without explicit key
    referer = request.headers.get("referer", "")
    origin = request.headers.get("origin", "")
    host = request.headers.get("host", "")
    if not token and (host in referer or host in origin or "vercel.app" in referer or "axiogen" in referer or "localhost" in host):
        return CONFIG.admin_key

    if not token or token == CONFIG.admin_key:
        return CONFIG.admin_key

    keys = load_api_keys()
    if token in keys:
        return token

    raise HTTPException(status_code=401, detail={"error": {"message": "Invalid API Key"}})

class SpeechRequest(BaseModel):
    model: str = Field(default="axiogen-v2")
    input: str = Field(..., max_length=CONFIG.max_input_length)
    voice: Optional[str] = Field(default=None)
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
    stream: bool = False
    format: str = Field(default="wav", description="Audio format: 'wav' or 'pcm'")

# ── WEBSOCKET REAL-TIME STREAMING ENDPOINT ──
@app.websocket("/ws/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    """
    WebSocket Streaming TTS:
    Supports:
    1. Full paragraph streaming: Client sends {"text": "...", "voice": "...", "speed": 1.0}
    2. Incremental Voice-Agent streaming: Client sends {"type": "token", "data": "word "} -> buffered -> synthesized per sentence.
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    session_id = f"ws_{int(time.time()*1000)}"
    print(f"[{session_id}] WebSocket client connected from {client_host}")

    incremental_buffer = IncrementalSentenceBuffer()

    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                data = json.loads(raw_msg)
            except Exception:
                await websocket.send_json({"type": "error", "message": "Invalid JSON format"})
                continue

            msg_type = data.get("type", "generate")
            voice = data.get("voice", CONFIG.default_voice)
            speed = float(data.get("speed", CONFIG.default_speed))
            out_format = data.get("format", "wav").lower()  # "pcm" or "wav"

            # ── MODE 1: Standard Direct Text Streaming ──
            if msg_type == "generate" or "text" in data:
                text = (data.get("text") or "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "message": "Empty text provided"})
                    continue

                await websocket.send_json({"type": "start", "session_id": session_id})
                total_dur_ms = 0.0
                sent_count = 0

                async for chunk in engine.stream_pipeline(text, voice=voice, speed=speed, request_id=session_id):
                    sent_count += 1
                    total_dur_ms += chunk.duration_ms
                    
                    audio_payload = chunk.wav_bytes if out_format == "wav" else chunk.pcm_bytes

                    # Send audio header frame
                    await websocket.send_json({
                        "type": "audio_meta",
                        "sentence_index": chunk.sentence_idx,
                        "text": chunk.text,
                        "duration_ms": round(chunk.duration_ms, 1),
                        "gen_time_ms": round(chunk.gen_time_ms, 1),
                        "sample_rate": chunk.sample_rate,
                        "is_final": chunk.is_final
                    })
                    # Send binary audio chunk
                    await websocket.send_bytes(audio_payload)

                # Final completion frame
                await websocket.send_json({
                    "type": "done",
                    "total_sentences": sent_count,
                    "total_duration_ms": round(total_dur_ms, 1)
                })

            # ── MODE 2: Incremental Token/Sentence Voice-Agent Streaming ──
            elif msg_type == "token":
                token = data.get("data", "")
                completed_sentences = incremental_buffer.add_token(token)

                for sentence in completed_sentences:
                    async for chunk in engine.stream_pipeline(sentence, voice=voice, speed=speed, request_id=session_id):
                        audio_payload = chunk.wav_bytes if out_format == "wav" else chunk.pcm_bytes
                        await websocket.send_json({
                            "type": "audio_meta",
                            "text": chunk.text,
                            "duration_ms": round(chunk.duration_ms, 1),
                            "gen_time_ms": round(chunk.gen_time_ms, 1)
                        })
                        await websocket.send_bytes(audio_payload)

            elif msg_type == "flush":
                remaining = incremental_buffer.flush()
                if remaining:
                    async for chunk in engine.stream_pipeline(remaining, voice=voice, speed=speed, request_id=session_id):
                        audio_payload = chunk.wav_bytes if out_format == "wav" else chunk.pcm_bytes
                        await websocket.send_json({
                            "type": "audio_meta",
                            "text": chunk.text,
                            "duration_ms": round(chunk.duration_ms, 1),
                            "is_final": True
                        })
                        await websocket.send_bytes(audio_payload)
                
                await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        print(f"[{session_id}] WebSocket client disconnected.")
    except Exception as e:
        print(f"[{session_id}] WebSocket error: {e}")

# ── REST API ENDPOINTS ──

@app.get("/", response_class=HTMLResponse)
@app.get("/playground", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h1>Axiogen Voice API</h1><p>Visit /health, /metrics, or /v1/voices</p>")

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "version": "2.0.0",
        "engine": "axiogen-v2-streaming",
        "voices_loaded": len(engine.available_voices),
        "uptime_seconds": int(time.time() - START_TIME),
        "streaming_ws": "/ws/tts",
        "sample_rate": CONFIG.sample_rate
    }

@app.get("/metrics")
async def metrics(format: str = Query("json", description="'json' or 'prometheus'")):
    if format.lower() == "prometheus":
        return PlainTextResponse(GLOBAL_METRICS.get_prometheus_metrics())
    return GLOBAL_METRICS.get_summary()

@app.get("/v1/models")
async def list_models(_: str = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": "axiogen-v2",
                "object": "model",
                "created": int(START_TIME),
                "owned_by": "axiogen"
            }
        ]
    }

@app.get("/v1/voices")
async def list_voices(_: str = Depends(verify_api_key)):
    """Dynamically parses and metadata-enriches all voices discovered in the model."""
    voices_data = []
    for v in engine.available_voices:
        parts = v.split("_")
        gender = "Female" if parts[0].endswith("f") else "Male" if parts[0].endswith("m") else "Neutral"
        accent = "American" if parts[0].startswith("a") else "British" if parts[0].startswith("b") else "International"
        name = parts[1].capitalize() if len(parts) > 1 else v

        voices_data.append({
            "voice_id": v,
            "name": name,
            "accent": accent,
            "gender": gender,
            "style": "Neural Studio"
        })

    return {"voices": voices_data}

async def http_stream_generator(text: str, voice: Optional[str], speed: Optional[float]) -> AsyncGenerator[bytes, None]:
    """Generates framed binary WAV chunks for HTTP streaming responses."""
    async for chunk in engine.stream_pipeline(text, voice=voice, speed=speed):
        yield struct.pack(">I", len(chunk.wav_bytes)) + chunk.wav_bytes

@app.post("/v1/audio/speech")
@app.post("/v1/tts")
async def create_speech(request: SpeechRequest, _: str = Depends(verify_api_key)):
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    # If stream requested, stream sentence chunks progressively
    if request.stream:
        return StreamingResponse(
            http_stream_generator(request.input, request.voice, request.speed),
            media_type="application/octet-stream"
        )

    # Standard full WAV generation (synthesizes sentences progressively and merges)
    all_pcm = []
    async for chunk in engine.stream_pipeline(request.input, voice=request.voice, speed=request.speed):
        all_pcm.append(chunk.pcm_bytes)

    if not all_pcm:
        raise HTTPException(status_code=500, detail="Audio generation failed")

    # Combine raw PCM into single WAV
    import soundfile as sf
    import io
    import numpy as np
    raw_bytes = b"".join(all_pcm)
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    buf = io.BytesIO()
    sf.write(buf, samples, CONFIG.sample_rate, format='WAV', subtype='PCM_16')

    return Response(content=buf.getvalue(), media_type="audio/wav")

@app.post("/v1/keys/create")
async def create_key(request: dict = {}, _: str = Depends(verify_api_key)):
    name = request.get("name", "Untitled Key") if isinstance(request, dict) else "Untitled Key"
    new_key = f"axg_{secrets.token_hex(16)}"
    keys = load_api_keys()
    keys[new_key] = {"name": name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save_api_keys(keys)
    return {"key": new_key, "name": name}

@app.get("/v1/keys/list")
async def list_keys(_: str = Depends(verify_api_key)):
    keys = load_api_keys()
    masked_keys = []
    for k, v in keys.items():
        masked = k[:8] + "•" * 12 + k[-4:]
        masked_keys.append({"key": k, "masked": masked, "name": v.get("name", "Unnamed"), "created": v.get("created_at", "")})
    return {"keys": masked_keys}

@app.delete("/v1/keys/delete")
async def delete_key(key: str = Query(...), _: str = Depends(verify_api_key)):
    keys = load_api_keys()
    if key in keys:
        del keys[key]
        save_api_keys(keys)
        return {"message": "Key revoked"}
    raise HTTPException(status_code=404, detail="Key not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=CONFIG.host, port=CONFIG.port)
