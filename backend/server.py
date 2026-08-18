import os
import json
import time
import secrets
import asyncio
import io
import urllib.request
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

# Model paths and URLs
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models" if os.path.exists("/app") else os.path.join(os.path.dirname(__file__), "models"))
MODEL_PATH = os.path.join(MODELS_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(MODELS_DIR, "voices-v1.0.bin")

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

engine = None
available_voices_list = []

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
        try:
            with open(API_KEYS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_api_keys(keys):
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=4)

def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model file from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model file downloaded.")
    if not os.path.exists(VOICES_PATH):
        print(f"Downloading voices binary from {VOICES_URL}...")
        urllib.request.urlretrieve(VOICES_URL, VOICES_PATH)
        print("Voices binary downloaded.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, available_voices_list
    
    # Auto download models on startup if missing
    ensure_models()
    
    # Optimize ONNX Session for HF Spaces CPU
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = min(os.cpu_count() or 2, 4)
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    try:
        session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=['CPUExecutionProvider'])
        engine = Kokoro.from_session(session, VOICES_PATH)
    except Exception:
        engine = Kokoro(MODEL_PATH, VOICES_PATH)

    try:
        available_voices_list = list(engine.get_voices())
    except Exception:
        available_voices_list = list(VOICE_METADATA_MAP.keys())

    print(f"Axiogen Voice Engine v2 loaded with {len(available_voices_list)} voices.")
    yield
    engine = None

app = FastAPI(title="Axiogen Voice API", lifespan=lifespan, docs_url=None, redoc_url=None)

# Allow all origins for Vercel and external apps
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
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1].strip()
    elif authorization:
        token = authorization.strip()
    elif x_api_key:
        token = x_api_key.strip()
    elif api_key:
        token = api_key.strip()

    # Browser UI / same host auto-pass
    referer = request.headers.get("referer", "")
    origin = request.headers.get("origin", "")
    host = request.headers.get("host", "")
    if not token and (host in referer or host in origin or "vercel.app" in referer or "axiogen" in referer):
        return ADMIN_KEY

    if not token or token == ADMIN_KEY:
        return ADMIN_KEY

    keys = load_api_keys()
    if token in keys:
        return token

    raise HTTPException(status_code=401, detail={"error": {"message": "Invalid API Key"}})

async def generate_audio_wav(text: str, voice: str, speed: float) -> bytes:
    if engine is None:
        raise HTTPException(status_code=503, detail="Axiogen Voice Engine v2 is initializing...")

    speed = max(0.25, min(speed, 4.0))
    v = voice if voice in available_voices_list else "af_bella"
    lang = "en-gb" if v.startswith("b") else "en-us"

    try:
        samples, sample_rate = await asyncio.to_thread(engine.create, text, voice=v, speed=speed, lang=lang)
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

class SpeechRequest(BaseModel):
    model: str = Field(default="axiogen-v2")
    input: str = Field(..., max_length=5000)
    voice: str = Field(default="af_bella")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

# Endpoints
@app.get("/health")
async def health():
    return {
        "status": "operational",
        "version": "2.0.0",
        "engine": "axiogen-v2",
        "voices_loaded": len(available_voices_list),
        "uptime_seconds": int(time.time() - START_TIME),
        "streaming": True
    }

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
    voices = available_voices_list or list(VOICE_METADATA_MAP.keys())
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
@app.post("/v1/tts")
async def create_speech(request: SpeechRequest, _: str = Depends(verify_api_key)):
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    wav_bytes = await generate_audio_wav(request.input, request.voice, request.speed)
    return Response(content=wav_bytes, media_type="audio/wav")

@app.post("/v1/tts/chunk")
async def create_chunk(request: dict, _: str = Depends(verify_api_key)):
    text = (request.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    voice = request.get("voice", "af_bella")
    speed = float(request.get("speed", 1.0))
    wav_bytes = await generate_audio_wav(text, voice, speed)
    return Response(content=wav_bytes, media_type="audio/wav")

@app.post("/v1/keys/create")
async def create_key(request: dict = {}, _: str = Depends(verify_api_key)):
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
async def list_keys(_: str = Depends(verify_api_key)):
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
async def delete_key(key: str = Query(...), _: str = Depends(verify_api_key)):
    keys = load_api_keys()
    if key in keys:
        del keys[key]
        save_api_keys(keys)
        return {"message": "Key revoked"}
    raise HTTPException(status_code=404, detail="Key not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860)
