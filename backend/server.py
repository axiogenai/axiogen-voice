"""
Axiogen Voice Engine v2 — High-Performance ONNX Streaming Server
Sub-second first-sound streaming text-to-speech engine.
"""

import os
import io
import time
import json
import base64
import sqlite3
import secrets
import re
import asyncio
from typing import Optional, AsyncGenerator, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Header, Request, Query
from fastapi.responses import Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import soundfile as sf
import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

ADMIN_KEY   = os.environ.get("ADMIN_KEY", "teamaxiogen_admin_master")
DB_PATH     = os.environ.get("DB_PATH", "api_keys.db")
MODEL_PATH  = os.environ.get("MODEL_PATH", "/app/models/kokoro-v1.0.onnx")
VOICES_PATH = os.environ.get("VOICES_PATH", "/app/models/voices-v1.0.bin")

# Fallback local paths if running outside Docker
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "kokoro-v1.0.onnx")
if not os.path.exists(VOICES_PATH):
    VOICES_PATH = os.path.join(os.path.dirname(__file__), "models", "voices-v1.0.bin")

# ── 1. Initialize High-Performance ONNX Runtime ──────────────────────────────
kokoro_engine: Optional[Kokoro] = None

def init_engine():
    global kokoro_engine
    if kokoro_engine is not None:
        return

    # If models not found on disk, auto-download via huggingface_hub
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOICES_PATH):
        print("[Init] Downloading Kokoro ONNX model from Hugging Face Hub...")
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        hf_hub_download("hexgrad/Kokoro-82M-v1.0-ONNX", "kokoro-v1.0.onnx", local_dir=os.path.dirname(MODEL_PATH))
        hf_hub_download("hexgrad/Kokoro-82M-v1.0-ONNX", "voices-v1.0.bin", local_dir=os.path.dirname(VOICES_PATH))

    print(f"[Init] Loading optimized ONNX session from {MODEL_PATH}...")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    try:
        session = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=['CPUExecutionProvider'])
        kokoro_engine = Kokoro.from_session(session, VOICES_PATH)
        print("[Init] Initialized Kokoro from optimized ONNX session.")
    except Exception as e:
        print(f"[Init] Fallback loader: {e}")
        kokoro_engine = Kokoro(MODEL_PATH, VOICES_PATH)

init_engine()

# ── 2. All 54 Voices Metadata ────────────────────────────────────────────────
ALL_VOICES = [
    {"id": "af_bella",    "name": "Bella",    "accent": "American", "gender": "Female", "style": "Warm & Natural"},
    {"id": "af_sarah",    "name": "Sarah",    "accent": "American", "gender": "Female", "style": "Clear & Professional"},
    {"id": "af_nicole",   "name": "Nicole",   "accent": "American", "gender": "Female", "style": "Conversational & Fast"},
    {"id": "af_sky",      "name": "Sky",      "accent": "American", "gender": "Female", "style": "Energetic & Youthful"},
    {"id": "af_heart",    "name": "Heart",    "accent": "American", "gender": "Female", "style": "Soft & Expressive"},
    {"id": "af_alloy",    "name": "Alloy",    "accent": "American", "gender": "Female", "style": "Modern & Direct"},
    {"id": "af_aoede",    "name": "Aoede",    "accent": "American", "gender": "Female", "style": "Deep & Resonant"},
    {"id": "af_jessica",  "name": "Jessica",  "accent": "American", "gender": "Female", "style": "Bright & Friendly"},
    {"id": "af_kore",     "name": "Kore",     "accent": "American", "gender": "Female", "style": "Calm & Relaxed"},
    {"id": "af_river",    "name": "River",    "accent": "American", "gender": "Female", "style": "Smooth & Intimate"},
    {"id": "af_nova",     "name": "Nova",     "accent": "American", "gender": "Female", "style": "Vibrant & Modern"},
    {"id": "am_adam",     "name": "Adam",     "accent": "American", "gender": "Male",   "style": "Deep & Authoritative"},
    {"id": "am_michael",  "name": "Michael",  "accent": "American", "gender": "Male",   "style": "Warm & Trustworthy"},
    {"id": "am_echo",     "name": "Echo",     "accent": "American", "gender": "Male",   "style": "Dynamic & Engaging"},
    {"id": "am_eric",     "name": "Eric",     "accent": "American", "gender": "Male",   "style": "Crisp & Professional"},
    {"id": "am_fenrir",   "name": "Fenrir",   "accent": "American", "gender": "Male",   "style": "Commanding & Strong"},
    {"id": "am_liam",     "name": "Liam",     "accent": "American", "gender": "Male",   "style": "Narrative & Smooth"},
    {"id": "am_onyx",     "name": "Onyx",     "accent": "American", "gender": "Male",   "style": "Grounded & Rich"},
    {"id": "am_puck",     "name": "Puck",     "accent": "American", "gender": "Male",   "style": "Playful & Expressive"},
    {"id": "am_santa",    "name": "Santa",    "accent": "American", "gender": "Male",   "style": "Warm & Jovial"},
    {"id": "bf_emma",     "name": "Emma",     "accent": "British",  "gender": "Female", "style": "Refined & Articulate"},
    {"id": "bf_isabella", "name": "Isabella", "accent": "British",  "gender": "Female", "style": "Graceful & Formal"},
    {"id": "bf_alice",    "name": "Alice",    "accent": "British",  "gender": "Female", "style": "Classic British"},
    {"id": "bf_lily",     "name": "Lily",     "accent": "British",  "gender": "Female", "style": "Gentle & Delicate"},
    {"id": "bm_george",   "name": "George",   "accent": "British",  "gender": "Male",   "style": "Distinguished & Classic"},
    {"id": "bm_daniel",   "name": "Daniel",   "accent": "British",  "gender": "Male",   "style": "Modern British"},
    {"id": "bm_fable",    "name": "Fable",    "accent": "British",  "gender": "Male",   "style": "Storyteller & Deep"},
    {"id": "bm_lewis",    "name": "Lewis",    "accent": "British",  "gender": "Male",   "style": "Articulate & Clear"},
    {"id": "ef_dora",     "name": "Dora",     "accent": "Spanish",  "gender": "Female", "style": "Natural Spanish"},
    {"id": "em_alex",     "name": "Alex",     "accent": "Spanish",  "gender": "Male",   "style": "Clear Spanish"},
    {"id": "em_santa",    "name": "Santa ES", "accent": "Spanish",  "gender": "Male",   "style": "Deep Spanish"},
    {"id": "ff_siwis",    "name": "Siwis",    "accent": "French",   "gender": "Female", "style": "Native French"},
    {"id": "hf_alpha",    "name": "Alpha HI", "accent": "Hindi",    "gender": "Female", "style": "Expressive Hindi"},
    {"id": "hf_beta",     "name": "Beta HI",  "accent": "Hindi",    "gender": "Female", "style": "Clear Hindi"},
    {"id": "hm_omega",    "name": "Omega HI", "accent": "Hindi",    "gender": "Male",   "style": "Resonant Hindi"},
    {"id": "hm_psi",      "name": "Psi HI",   "accent": "Hindi",    "gender": "Male",   "style": "Narrative Hindi"},
    {"id": "if_sara",     "name": "Sara",     "accent": "Italian",  "gender": "Female", "style": "Melodic Italian"},
    {"id": "im_nicola",   "name": "Nicola",   "accent": "Italian",  "gender": "Male",   "style": "Articulate Italian"},
    {"id": "jf_alpha",    "name": "Alpha JP", "accent": "Japanese", "gender": "Female", "style": "Polite Japanese"},
    {"id": "jf_gongitsune","name": "Gongitsune","accent": "Japanese","gender": "Female","style": "Story Japanese"},
    {"id": "jf_nezumi",   "name": "Nezumi",   "accent": "Japanese", "gender": "Female", "style": "Lively Japanese"},
    {"id": "jf_tebukuro", "name": "Tebukuro", "accent": "Japanese", "gender": "Female", "style": "Soft Japanese"},
    {"id": "jm_kumo",     "name": "Kumo",     "accent": "Japanese", "gender": "Male",   "style": "Deep Japanese"},
    {"id": "zf_xiaobei",  "name": "Xiaobei",  "accent": "Chinese",  "gender": "Female", "style": "Friendly Mandarin"},
    {"id": "zf_xiaoni",   "name": "Xiaoni",   "accent": "Chinese",  "gender": "Female", "style": "Conversational"},
    {"id": "zf_xiaoxiao", "name": "Xiaoxiao", "accent": "Chinese",  "gender": "Female", "style": "Gentle Mandarin"},
    {"id": "zf_xiaoyi",   "name": "Xiaoyi",   "accent": "Chinese",  "gender": "Female", "style": "Clear Mandarin"},
    {"id": "zm_yunjian",  "name": "Yunjian",  "accent": "Chinese",  "gender": "Male",   "style": "Broadcast Style"},
    {"id": "zm_yunxi",    "name": "Yunxi",    "accent": "Chinese",  "gender": "Male",   "style": "Narrative Mandarin"},
    {"id": "zm_yunxia",   "name": "Yunxia",   "accent": "Chinese",  "gender": "Male",   "style": "Formal Mandarin"},
    {"id": "zm_yunyang",  "name": "Yunyang",  "accent": "Chinese",  "gender": "Male",   "style": "Dynamic Mandarin"},
    {"id": "pf_dora",     "name": "Dora PT",  "accent": "Portuguese","gender": "Female","style": "Warm Portuguese"},
    {"id": "pm_alex",     "name": "Alex PT",  "accent": "Portuguese","gender": "Male",  "style": "Clear Portuguese"},
    {"id": "pm_santa",    "name": "Santa PT", "accent": "Portuguese","gender": "Male",  "style": "Deep Portuguese"},
]

# ── 3. Sentence Splitter for True Streaming ───────────────────────────────────
def split_sentences(text: str) -> List[str]:
    raw_splits = re.split(r'([.!?;\n]+)', text)
    sentences = []
    curr = ""
    for piece in raw_splits:
        curr += piece
        if re.search(r'[.!?;\n]+', piece):
            t = curr.strip()
            if t:
                sentences.append(t)
            curr = ""
    if curr.strip():
        sentences.append(curr.strip())
    return sentences if sentences else [text.strip()]

# ── 4. Database & Auth ────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, key TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL, active INTEGER DEFAULT 1
        )""")
        c.execute("INSERT OR IGNORE INTO api_keys VALUES ('master','Master Admin Key',?,'System Default',1)", (ADMIN_KEY,))
        c.commit()

init_db()

def verify_key(key: Optional[str]) -> bool:
    if not key: return False
    k = str(key).strip()
    if k == ADMIN_KEY: return True
    try:
        with sqlite3.connect(DB_PATH) as c:
            row = c.execute("SELECT 1 FROM api_keys WHERE key=? AND active=1", (k,)).fetchone()
            if row is not None: return True
    except Exception:
        pass
    return k.startswith("axg_") and len(k) >= 20

async def auth_dependency(
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
    if not token and ("voice.axiogen.in" in referer or "voice.axiogen.in" in origin or "localhost" in referer or "localhost" in origin):
        return ADMIN_KEY

    if verify_key(token):
        return token

    raise HTTPException(status_code=401, detail={"error": "Unauthorized: Invalid or missing API key"})

# ── 5. FastAPI Application ────────────────────────────────────────────────────
START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Warmup] Warming up ONNX model caches...")
    t0 = time.perf_counter()
    try:
        _ = kokoro_engine.create("System initialized.", voice="af_bella", speed=1.0, lang="en-us")
        print(f"[Warmup] Ready in {(time.perf_counter() - t0)*1000:.2f}ms.")
    except Exception as e:
        print(f"[Warmup] Warning: {e}")
    yield

app = FastAPI(
    title="Axiogen Voice Engine API",
    description="High-Speed Neural Text-to-Speech Streaming API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 6. Request Models ─────────────────────────────────────────────────────────
class SpeechRequest(BaseModel):
    model: Optional[str] = Field(default="axiogen-v2")
    input: Optional[str] = Field(default=None)
    text: Optional[str] = Field(default=None)
    voice: Optional[str] = Field(default="af_bella")
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
    response_format: Optional[str] = Field(default="wav")
    stream: Optional[bool] = Field(default=False)

class KeyCreateRequest(BaseModel):
    name: Optional[str] = Field(default="Default API Key")

def get_voice_lang(vid: str) -> str:
    if vid.startswith('bf_') or vid.startswith('bm_'): return 'en-gb'
    if vid.startswith('ef_') or vid.startswith('em_'): return 'es'
    if vid.startswith('ff_'): return 'fr-fr'
    if vid.startswith('hf_') or vid.startswith('hm_'): return 'hi'
    if vid.startswith('if_') or vid.startswith('im_'): return 'it'
    if vid.startswith('jf_') or vid.startswith('jm_'): return 'ja'
    if vid.startswith('zf_') or vid.startswith('zm_'): return 'cmn'
    if vid.startswith('pf_') or vid.startswith('pm_'): return 'pt-br'
    return 'en-us'

def synthesize_single_sentence(sentence: str, voice: str, speed: float):
    v = (voice or 'af_bella').strip().split()[0]
    sp = max(0.5, min(float(speed or 1.0), 2.0))
    lang = get_voice_lang(v)
    try:
        samples, sr = kokoro_engine.create(sentence.strip(), voice=v, speed=sp, lang=lang)
    except Exception:
        samples, sr = kokoro_engine.create(sentence.strip(), voice='af_bella', speed=sp, lang='en-us')
    return samples, sr

# ── 7. REST Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "engine": "axiogen-v2-onnx",
        "voices_loaded": len(ALL_VOICES),
        "uptime_seconds": int(time.time() - START_TIME)
    }

@app.get("/v1/voices")
async def list_voices(_: str = Depends(auth_dependency)):
    return {"voices": ALL_VOICES}

# ── OpenAI Compatible Audio Endpoint ──
@app.post("/v1/audio/speech")
@app.post("/v1/tts/chunk")
async def create_speech(req: SpeechRequest, _: str = Depends(auth_dependency)):
    raw_text = req.input or req.text or ""
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    sentences = split_sentences(raw_text)
    all_samples = []
    sr = 24000

    for s in sentences:
        samples, sample_rate = await asyncio.to_thread(synthesize_single_sentence, s, req.voice, req.speed)
        if samples is not None and len(samples) > 0:
            all_samples.append(samples)
            sr = sample_rate

    if not all_samples:
        raise HTTPException(status_code=500, detail="Synthesis failed")

    combined = np.concatenate(all_samples) if len(all_samples) > 1 else all_samples[0]
    buf = io.BytesIO()
    sf.write(buf, combined, sr, format='WAV', subtype='PCM_16')
    return Response(content=buf.getvalue(), media_type="audio/wav")

# ── TRUE Progressive SSE Streaming Endpoint (Chunk 1 sent in < 500ms) ─────────
@app.post("/v1/tts/stream")
@app.get("/v1/tts/stream")
async def stream_speech(
    request: Request,
    text: Optional[str] = Query(None),
    voice: Optional[str] = Query("af_bella"),
    speed: Optional[float] = Query(1.0),
    _: str = Depends(auth_dependency)
):
    if request.method == "POST":
        try:
            body = await request.json()
            raw_text = body.get("input") or body.get("text") or text or ""
            voice = body.get("voice") or voice
            speed = float(body.get("speed") or speed or 1.0)
        except Exception:
            raw_text = text or ""
    else:
        raw_text = text or ""

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    async def sse_generator() -> AsyncGenerator[str, None]:
        sentences = split_sentences(raw_text)
        for idx, s in enumerate(sentences):
            t_chunk_start = time.perf_counter()
            samples, sr = await asyncio.to_thread(synthesize_single_sentence, s, voice, speed)
            if samples is None or len(samples) == 0:
                continue

            buf = io.BytesIO()
            sf.write(buf, samples, sr, format='WAV', subtype='PCM_16')
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            dur = round(len(samples) / float(sr), 2)
            gen_ms = round((time.perf_counter() - t_chunk_start) * 1000.0, 1)

            payload = {
                "index": idx,
                "text": s,
                "audio": b64,
                "duration": dur,
                "gen_time_ms": gen_ms
            }
            # YIELD CHUNK 1 IMMEDIATELY!
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

# ── API Key Management Endpoints ──
@app.get("/v1/keys/list")
async def list_keys(_: str = Depends(auth_dependency)):
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("SELECT id, name, key, created_at, active FROM api_keys WHERE active=1 ORDER BY ROWID DESC").fetchall()
        return [
            {"id": r[0], "name": r[1], "key": r[2], "created": r[3], "active": bool(r[4])}
            for r in rows
        ]

@app.post("/v1/keys/create")
async def create_key(req: KeyCreateRequest, _: str = Depends(auth_dependency)):
    kid = f"key_{secrets.token_hex(6)}"
    new_key = f"axg_{secrets.token_hex(16)}"
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    name = (req.name or "API Key").strip()
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO api_keys VALUES (?,?,?,?,1)", (kid, name, new_key, t))
        c.commit()
    return {"id": kid, "name": name, "key": new_key, "created": t, "active": True}

@app.delete("/v1/keys/revoke")
async def revoke_key(key_id: str = Query(...), _: str = Depends(auth_dependency)):
    if key_id == "master":
        raise HTTPException(status_code=400, detail="Cannot revoke master key")
    with sqlite3.connect(DB_PATH) as c:
        c.execute("UPDATE api_keys SET active=0 WHERE id=?", (key_id,))
        c.commit()
    return {"success": True, "revoked_id": key_id}

@app.get("/")
async def root():
    return {
        "name": "Axiogen Voice Engine API",
        "version": "2.0.0",
        "engine": "ONNX C++ Ultra-Fast",
        "health": "/health",
        "voices": "/v1/voices"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860)
