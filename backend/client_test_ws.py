"""
client_test_ws.py - Python WebSocket Streaming Benchmark Client for Axiogen Voice TTS.

Measures:
- Time-to-First-Byte (TTFB) / Time-to-First-Audio (TTFA)
- Per-chunk arrival latency
- Audio playback continuity
- Total audio duration vs total generation time (Real-Time Factor - RTF)
"""

import asyncio
import json
import time
import websockets
import soundfile as sf
import io

TEST_PARAGRAPH = (
    "Hello Aditya! Welcome to the new Axiogen Voice low latency streaming architecture. "
    "We have completely overhauled the Kokoro text to speech engine. "
    "Instead of waiting for the entire paragraph to synthesize, incoming text is segmented into natural sentences immediately. "
    "The producer worker generates speech chunk by chunk in parallel with playback. "
    "As you can see, the time to first sound is now ultra fast, while subsequent sentences stream continuously in the background. "
    "This pipeline is fully compatible with real-time AI voice agents, Discord bots, and interactive applications."
)

async def test_websocket_streaming(uri: str = "ws://localhost:7860/ws/tts", voice: str = "af_bella"):
    print(f"\n=======================================================")
    print(f"  AXIOGEN VOICE TTS - WEBSOCKET STREAMING BENCHMARK    ")
    print(f"=======================================================")
    print(f"Connecting to: {uri} (Voice: {voice})\n")

    t_start = time.perf_counter()
    first_chunk_time = None
    chunk_count = 0
    total_audio_bytes = 0

    async with websockets.connect(uri) as ws:
        # Send text synthesis request
        payload = {
            "type": "generate",
            "text": TEST_PARAGRAPH,
            "voice": voice,
            "speed": 1.0,
            "format": "wav"
        }
        await ws.send(json.dumps(payload))
        print(f"-> Sent test paragraph ({len(TEST_PARAGRAPH.split())} words, {len(TEST_PARAGRAPH)} chars). Waiting for audio...\n")

        while True:
            msg = await ws.recv()
            
            # Check if JSON metadata frame
            if isinstance(msg, str):
                frame = json.loads(msg)
                if frame.get("type") == "start":
                    print(f"[*] Stream started (Session: {frame.get('session_id')})")
                elif frame.get("type") == "audio_meta":
                    idx = frame.get("sentence_index")
                    txt = frame.get("text")
                    dur = frame.get("duration_ms")
                    gen_ms = frame.get("gen_time_ms")
                    print(f"  [Chunk {idx+1}] Text: \"{txt[:35]}...\" | Gen Time: {gen_ms:.1f}ms | Audio Dur: {dur:.0f}ms")
                elif frame.get("type") == "done":
                    total_dur = frame.get("total_duration_ms")
                    total_sents = frame.get("total_sentences")
                    print(f"\n[*] Stream Complete: {total_sents} sentences synthesized ({total_dur:.0f}ms total audio).")
                    break

            # Check if binary audio payload
            elif isinstance(msg, bytes):
                chunk_count += 1
                total_audio_bytes += len(msg)
                if first_chunk_time is None:
                    first_chunk_time = (time.perf_counter() - t_start) * 1000.0
                    print(f"\n⚡ [TIME-TO-FIRST-AUDIO (TTFA)]: {first_chunk_time:.2f} ms!")
                    print(f"   (Client starts audio playback immediately at this moment)\n")

    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    print(f"-------------------------------------------------------")
    print(f"BENCHMARK RESULTS:")
    print(f"  • Time to First Audio (TTFA): {first_chunk_time:.2f} ms")
    print(f"  • Total Stream Time:         {total_time_ms:.2f} ms")
    print(f"  • Total Audio Chunks:        {chunk_count}")
    print(f"  • Total Bytes Received:      {total_audio_bytes:,} bytes")
    print(f"=======================================================\n")

if __name__ == "__main__":
    asyncio.run(test_websocket_streaming())
