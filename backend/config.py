"""
config.py - Centralized, environment-driven configuration for Axiogen Voice TTS Server.
All settings can be dynamically overridden via environment variables or .env file.
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ServerConfig:
    # Server network settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "7860"))
    admin_key: str = os.getenv("ADMIN_KEY", "teamaxiogen_admin_master")
    api_keys_file: str = os.getenv("API_KEYS_FILE", "api_keys.json")
    cors_origins: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))

    # Model & filesystem paths
    models_dir: str = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))
    model_filename: str = os.getenv("MODEL_FILENAME", "kokoro-v1.0.onnx")
    voices_filename: str = os.getenv("VOICES_FILENAME", "voices-v1.0.bin")
    
    # Model download URLs
    model_url: str = os.getenv(
        "MODEL_DOWNLOAD_URL", 
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    )
    voices_url: str = os.getenv(
        "VOICES_DOWNLOAD_URL",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    )

    # Audio & inference parameters
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "24000"))
    default_voice: str = os.getenv("DEFAULT_VOICE", "af_bella")
    default_speed: float = float(os.getenv("DEFAULT_SPEED", "1.0"))
    max_input_length: int = int(os.getenv("MAX_INPUT_LENGTH", "10000"))
    
    # Queue & streaming concurrency
    queue_maxsize: int = int(os.getenv("TTS_QUEUE_MAXSIZE", "8"))
    max_sentence_words: int = int(os.getenv("TTS_MAX_SENTENCE_WORDS", "25"))
    num_threads: int = int(os.getenv("ORT_INTRA_OP_NUM_THREADS", str(min(os.cpu_count() or 4, 8))))
    enable_warmup: bool = os.getenv("ENABLE_WARMUP", "true").lower() in ("true", "1", "yes")

    @property
    def model_path(self) -> str:
        return os.path.join(self.models_dir, self.model_filename)

    @property
    def voices_path(self) -> str:
        return os.path.join(self.models_dir, self.voices_filename)

CONFIG = ServerConfig()
