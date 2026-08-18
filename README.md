# ⚡ Axiogen Voice Platform

> High-performance Neural Speech & Text-to-Speech (TTS) Platform by Axiogen AI.

[![Platform](https://img.shields.io/badge/Platform-Axiogen%20Voice-8B5CF6?style=for-the-badge&logo=fastapi)](https://voice.axiogen.in)
[![Deployment](https://img.shields.io/badge/Deployment-Vercel%20%2B%20API-10B981?style=for-the-badge&logo=vercel)](https://voice.axiogen.in)
[![License](https://img.shields.io/badge/License-MIT-38BDF8?style=for-the-badge)](LICENSE)

---

## 🌟 Overview

**Axiogen Voice** is an enterprise-grade AI speech synthesis platform featuring:
- 🎙️ **54+ High-Fidelity Neural Voices** (American, British, and Global accents).
- ⚡ **Zero-Wait Micro-Chunk Streaming** with client-side Web Audio API queueing.
- 🔑 **Built-in API Key Management** with `teamaxiogen_` & `axg_` security.
- 🔌 **Drop-in OpenAI SDK Compatibility** (`POST /v1/audio/speech`).
- 🖥️ **Studio Web Playground** with real-time audio scrubbing & wave animations.

---

## 📁 Repository Structure

```
axiogen-voice/
├── frontend/             # Vercel-ready Web Studio & Playground
│   ├── index.html        # Modern Dark UI (Playground, API Keys, Docs)
│   ├── vercel.json       # Vercel headers & routing configuration
│   └── package.json      # Node deployment manifest
└── backend/              # Python FastAPI & ONNX Neural Engine
    ├── server.py         # Multi-threaded REST & Streaming Server
    ├── Dockerfile        # Container build for Cloud/HF Spaces
    ├── requirements.txt  # Python dependencies
    └── README.md         # Backend API documentation
```

---

## 🚀 Quick Start (API)

### Python (OpenAI SDK Drop-in)
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://voice.axiogen.in/v1",
    api_key="teamaxiogen_admin_master"
)

response = client.audio.speech.create(
    model="axiogen-v2",
    voice="af_bella",
    input="Welcome to Axiogen Voice! Neural speech in real-time."
)
response.stream_to_file("output.wav")
```

### cURL
```bash
curl -X POST "https://voice.axiogen.in/v1/audio/speech" \
  -H "Authorization: Bearer teamaxiogen_admin_master" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "axiogen-v2",
    "input": "Hello from Axiogen Voice!",
    "voice": "af_bella",
    "speed": 1.0
  }' \
  --output speech.wav
```

---

## 🌐 Deploy to Vercel

1. Push this repository to GitHub: `https://github.com/axiogenai/axiogen-voice`
2. Connect to [Vercel](https://vercel.com)
3. Set **Root Directory** to `frontend`
4. Click **Deploy**!
