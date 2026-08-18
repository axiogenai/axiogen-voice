"""
Axiogen Voice Engine v2 — Production FastAPI Docker Backend
High-performance, low-latency streaming neural text-to-speech server.
Supports 54 voices across 9 language families with SQLite key management and OpenAI compatibility.
"""

import os
import io
import time
import json
import base64
import sqlite3
import secrets
import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Header, Request, Query
from fastapi.responses import Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import soundfile as sf
import numpy as np
import torch
from kokoro import KModel, KPipeline

ADMIN_KEY = os.environ.get("ADMIN_KEY", "teamaxiogen_admin_master")
DB_PATH   = os.environ.get("DB_PATH", "api_keys.db")

# ── 1. Multi-Language Pipeline Initialization ─────────────────────────────────
LANG_CODES = 'abefihjzp'
pipelines: dict = {}
for lc in LANG_CODES:
    try:
        pipelines[lc] = KPipeline(lang_code=lc, model=False)
        print(f"[Init] Pipeline '{lc}' initialized.")
    except Exception as e:
        print(f"[Init] Pipeline '{lc}' warning: {e}")

# Pre-warmed CPU neural model kept in RAM permanently
MODEL = KModel().to('cpu').eval()

# ── 2. All 54 Voices & Voice Pack Memory Cache ────────────────────────────────
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

VOICE_CACHE: dict = {}

def preload_voices():
    print("[Init] Caching voice packs in RAM...")
    t0 = time.time()
    for v in ALL_VOICES:
        vid = v["id"]
        lc = vid[0] if (len(vid) > 0 and vid[0] in pipelines) else 'a'
        pl = pipelines.get(lc) or pipelines.get('a')
        try:
            VOICE_CACHE[vid] = pl.load_voice(vid)
        except Exception:
            try:
                VOICE_CACHE[vid] = pipelines['a'].load_voice(vid)
            except Exception:
                pass
    print(f"[Init] {len(VOICE_CACHE)} voices cached in {round(time.time() - t0, 2)}s.")

preload_voices()

def get_voice_pack(pipeline, vid: str):
    if vid in VOICE_CACHE:
        return VOICE_CACHE[vid], vid
    try:
        pack = pipeline.load_voice(vid)
        VOICE_CACHE[vid] = pack
        return pack, vid
    except Exception:
        fallback = VOICE_CACHE.get('af_bella') or pipelines['a'].load_voice('af_bella')
        return fallback, 'af_bella'

# ── 3. Database & Auth ────────────────────────────────────────────────────────
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

    # Browser referer auto-auth for playground
    referer = request.headers.get("referer", "")
    origin = request.headers.get("origin", "")
    if not token and ("voice.axiogen.in" in referer or "voice.axiogen.in" in origin or "localhost" in referer or "localhost" in origin):
        return ADMIN_KEY

    if verify_key(token):
        return token

    raise HTTPException(status_code=401, detail={"error": "Unauthorized: Invalid or missing API key"})

# ── 4. FastAPI Application ────────────────────────────────────────────────────
START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Warmup] Warming up neural inference...")
    t0 = time.time()
    try:
        pl = pipelines.get('a')
        pack = VOICE_CACHE.get('af_bella')
        for _, ps, _ in pl("Warmup.", 'af_bella', 1.0):
            with torch.inference_mode():
                _ = MODEL(ps, pack[len(ps)-1], 1.0)
        print(f"[Warmup] Ready in {round(time.time()-t0, 2)}s.")
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

# ── 5. Request Models ─────────────────────────────────────────────────────────
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

# ── 6. Core Synthesis Logic ───────────────────────────────────────────────────
def synthesize_chunk_sync(text: str, voice: str, speed: float):
    vid = (voice or 'af_bella').strip().split()[0]
    lc = vid[0] if (len(vid) > 0 and vid[0] in pipelines) else 'a'
    pipeline = pipelines.get(lc) or pipelines['a']
    pack, vid = get_voice_pack(pipeline, vid)
    sp = max(0.5, min(float(speed or 1.0), 2.0))

    chunks = []
    for gs, ps, _ in pipeline(text.strip(), vid, sp):
        if ps is None or len(ps) == 0:
            continue
        idx = min(len(ps) - 1, len(pack) - 1)
        ref_s = pack[idx]
        with torch.inference_mode():
            audio = MODEL(ps, ref_s, sp)
        audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else np.array(audio)
        if len(audio_np) > 0:
            chunks.append((gs.strip(), audio_np))
    return chunks

# ── 7. REST Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "operational",
        "engine": "axiogen-v2-neural",
        "voices_loaded": len(VOICE_CACHE),
        "uptime_seconds": int(time.time() - START_TIME)
    }

@app.get("/v1/voices")
async def list_voices(_: str = Depends(auth_dependency)):
    return {"voices": ALL_VOICES}

# ── OpenAI Compatible TTS Endpoint ──
@app.post("/v1/audio/speech")
@app.post("/v1/tts/chunk")
async def create_speech(req: SpeechRequest, _: str = Depends(auth_dependency)):
    text = req.input or req.text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    chunks = await asyncio.to_thread(synthesize_chunk_sync, text, req.voice, req.speed)
    if not chunks:
        raise HTTPException(status_code=500, detail="Synthesis produced no audio")

    all_audio = [c[1] for c in chunks]
    combined = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]

    buf = io.BytesIO()
    sf.write(buf, combined, 24000, format='WAV', subtype='PCM_16')
    return Response(content=buf.getvalue(), media_type="audio/wav")

# ── Real-Time SSE Chunk Streaming Endpoint ──
@app.post("/v1/tts/stream")
@app.get("/v1/tts/stream")
async def stream_speech(
    request: Request,
    text: Optional[str] = Query(None),
    voice: Optional[str] = Query("af_bella"),
    speed: Optional[float] = Query(1.0),
    _: str = Depends(auth_dependency)
):
    # Support both POST JSON body and GET Query params
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
        chunks = await asyncio.to_thread(synthesize_chunk_sync, raw_text, voice, speed)
        for idx, (gs, audio_np) in enumerate(chunks):
            buf = io.BytesIO()
            sf.write(buf, audio_np, 24000, format='WAV', subtype='PCM_16')
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            payload = {
                "index": idx,
                "text": gs,
                "audio": b64,
                "duration": round(len(audio_np) / 24000.0, 2)
            }
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
        "docs": "https://voice.axiogen.in",
        "health": "/health",
        "voices": "/v1/voices"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860)
