import os
import json
import time
import secrets
import asyncio
import io
import soundfile as sf
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Header, Request, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import onnxruntime as ort
from kokoro_onnx import Kokoro

load_dotenv()

ADMIN_KEY = os.getenv("ADMIN_KEY", "teamaxiogen_admin_master")
API_KEYS_FILE = "api_keys.json"
START_TIME = time.time()

# Global TTS Engine instance
engine = None

VOICE_METADATA_MAP = {
    "af_bella": {"name": "Bella", "accent": "American", "gender": "Female", "style": "Warm & Natural"},
    "af_heart": {"name": "Heart", "accent": "American", "gender": "Female", "style": "Soft & Expressive"},
    "af_sarah": {"name": "Sarah", "accent": "American", "gender": "Female", "style": "Clear & Professional"},
    "af_nicole": {"name": "Nicole", "accent": "American", "gender": "Female", "style": "Bright & Friendly"},
    "af_sky": {"name": "Sky", "accent": "American", "gender": "Female", "style": "Youthful & Energetic"},
    "am_adam": {"name": "Adam", "accent": "American", "gender": "Male", "style": "Deep & Authoritative"},
    "am_michael": {"name": "Michael", "accent": "American", "gender": "Male", "style": "Warm & Trustworthy"},
    "bf_emma": {"name": "Emma", "accent": "British", "gender": "Female", "style": "Elegant & Refined"},
    "bf_isabella": {"name": "Isabella", "accent": "British", "gender": "Female", "style": "Graceful"},
    "bf_alice": {"name": "Alice", "accent": "British", "gender": "Female", "style": "Classic British"},
    "bm_george": {"name": "George", "accent": "British", "gender": "Male", "style": "Distinguished"},
    "bm_daniel": {"name": "Daniel", "accent": "British", "gender": "Male", "style": "Modern British"}
}

def load_api_keys():
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_api_keys(keys):
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=4)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Initialize api_keys.json if not present
    if not os.path.exists(API_KEYS_FILE):
        save_api_keys({})

    # Optimize ONNX Session for backend performance
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Define model paths (assumes Docker context downloading to /app/models)
    model_path = "/app/models/kokoro-v1.0.onnx"
    voices_path = "/app/models/voices-v1.0.bin"

    # Fallback paths for local development
    if not os.path.exists(model_path):
        model_path = "models/kokoro-v1.0.onnx"
        voices_path = "models/voices-v1.0.bin"

    if os.path.exists(model_path) and os.path.exists(voices_path):
        try:
            # Load with optimized options and fallback
            session = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
            try:
                engine = Kokoro.from_session(session, voices_path)
            except AttributeError:
                engine = Kokoro(model_path, voices_path)
            print("Axiogen Voice Engine v2 loaded successfully.")
        except Exception as e:
            print(f"Error loading engine: {e}")
    else:
        print("Warning: Engine model files not found. Axiogen Voice Engine v2 is not loaded.")

    yield

    engine = None

app = FastAPI(title="Axiogen Voice API", lifespan=lifespan, docs_url=None, redoc_url=None)

# Allow all origins for Vercel Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    # Browser Request Auto-Auth
    referer = request.headers.get("referer", "")
    origin = request.headers.get("origin", "")
    host = request.headers.get("host", "")

    # If the request originated from the same host, bypass auth with admin key
    if host and ((origin and host in origin) or (referer and host in referer)):
        return ADMIN_KEY

    # Check for tokens
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1]
    elif x_api_key:
        token = x_api_key
    elif api_key:
        token = api_key

    if not token:
        raise HTTPException(status_code=401, detail="Missing API Key")

    # Admin Key bypass
    if token == ADMIN_KEY:
        return token

    # Check valid API keys
    keys = load_api_keys()
    if token not in keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    return token

async def generate_audio_wav(text: str, voice: str, speed: float) -> bytes:
    if engine is None:
        raise HTTPException(status_code=503, detail="Axiogen Voice Engine v2 is not loaded")

    # Limit speed bounds
    speed = max(0.25, min(speed, 4.0))
    lang = "en-gb" if voice.startswith("b") else "en-us"

    try:
        # Offload CPU intensive engine execution to threadpool
        samples, sample_rate = await asyncio.to_thread(engine.create, text, voice=voice, speed=speed, lang=lang)
        
        # Serialize to WAV
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Request Models
class SpeechRequest(BaseModel):
    model: str = Field(default="axiogen-v2")
    input: str = Field(..., max_length=5000)
    voice: str = Field(...)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

class ChunkRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    voice: str = Field(...)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

class KeyCreateRequest(BaseModel):
    name: str

# Endpoints
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "engine": "axiogen-v2",
        "voices_loaded": engine is not None,
        "uptime_seconds": int(time.time() - START_TIME)
    }

@app.get("/v1/models")
async def list_models(token: str = Depends(verify_api_key)):
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
async def list_voices(token: str = Depends(verify_api_key)):
    voices = list(VOICE_METADATA_MAP.keys())
    if engine is not None:
        try:
            voices = list(engine.get_voices())
        except AttributeError:
            pass

    voices_data = []
    for v in voices:
        if v in VOICE_METADATA_MAP:
            meta = VOICE_METADATA_MAP[v]
        else:
            parts = v.split("_")
            gender = "Female" if parts[0].endswith("f") else "Male" if parts[0].endswith("m") else "Unknown"
            accent = "American" if parts[0].startswith("a") else "British" if parts[0].startswith("b") else "Unknown"
            name = parts[1].capitalize() if len(parts) > 1 else v
            meta = {"name": name, "accent": accent, "gender": gender, "style": "Standard"}

        voices_data.append({
            "voice_id": v,
            "name": meta["name"],
            "accent": meta["accent"],
            "gender": meta["gender"],
            "style": meta["style"],
        })

    return {"voices": voices_data}

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest, token: str = Depends(verify_api_key)):
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    wav_bytes = await generate_audio_wav(request.input, request.voice, request.speed)
    return Response(content=wav_bytes, media_type="audio/wav")

@app.post("/v1/tts/chunk")
async def create_chunk(request: dict, token: str = Depends(verify_api_key)):
    text = (request.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    voice = request.get("voice", "af_bella")
    speed = float(request.get("speed", 1.0))
    wav_bytes = await generate_audio_wav(text, voice, speed)
    return Response(content=wav_bytes, media_type="audio/wav")

@app.post("/v1/keys/create")
async def create_key(request: dict = {}, token: str = Depends(verify_api_key)):
    if token != ADMIN_KEY:
        # Also allow browser-originated requests
        pass
    
    name = request.get("name", "Untitled Key") if isinstance(request, dict) else "Untitled Key"
    new_key = f"axg_{secrets.token_hex(16)}"
    keys = load_api_keys()
    keys[new_key] = {
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    save_api_keys(keys)
    
    return {"key": new_key, "name": name}

@app.get("/v1/keys/list")
async def list_keys(token: str = Depends(verify_api_key)):
    keys = load_api_keys()
    masked_keys = []
    for k, v in keys.items():
        masked = k[:8] + "•" * 12 + k[-4:]
        masked_keys.append({
            "key": k,
            "masked": masked,
            "name": v.get("name", "Unnamed"),
            "created": v.get("created_at", "")
        })
        
    return {"keys": masked_keys}

@app.delete("/v1/keys/delete")
async def delete_key(key: str = Query(...), token: str = Depends(verify_api_key)):
    keys = load_api_keys()
    if key in keys:
        del keys[key]
        save_api_keys(keys)
        return {"message": "Key revoked"}
    raise HTTPException(status_code=404, detail="Key not found")

