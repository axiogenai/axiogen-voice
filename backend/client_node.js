/**
 * client_node.js - Node.js WebSocket Streaming Client for Axiogen Voice TTS.
 * 
 * Demonstrates:
 * - Real-time sentence audio streaming over /ws/tts
 * - Zero-latency first-sound decoding
 * - Output file streaming
 */

import WebSocket from 'ws';
import fs from 'fs';

const WS_URL = process.env.WS_URL || 'ws://localhost:7860/ws/tts';
const OUTPUT_FILE = 'streamed_output.wav';

const text = "Hello! Welcome to Axiogen Voice real-time WebSocket streaming. As you can hear, speech begins playing almost immediately!";

console.log(`Connecting to Axiogen Voice TTS at: ${WS_URL}...`);
const ws = new WebSocket(WS_URL);

let t0 = Date.now();
let firstAudioTime = null;
let chunkIndex = 0;
const audioChunks = [];

ws.on('open', () => {
  console.log('Connected! Sending text payload...');
  t0 = Date.now();
  ws.send(JSON.stringify({
    type: 'generate',
    text: text,
    voice: 'af_bella',
    speed: 1.0,
    format: 'wav'
  }));
});

ws.on('message', (data, isBinary) => {
  if (!isBinary) {
    const meta = JSON.parse(data.toString());
    if (meta.type === 'start') {
      console.log(`[*] Stream started (Session: ${meta.session_id})`);
    } else if (meta.type === 'audio_meta') {
      console.log(`[Meta] Chunk ${meta.sentence_index + 1}: "${meta.text}" (${meta.duration_ms}ms audio) in ${meta.gen_time_ms}ms`);
    } else if (meta.type === 'done') {
      console.log(`\n[*] Stream completed: ${meta.total_sentences} sentences (${meta.total_duration_ms}ms audio)`);
      ws.close();
    }
  } else {
    chunkIndex++;
    audioChunks.push(data);
    if (!firstAudioTime) {
      firstAudioTime = Date.now() - t0;
      console.log(`\n⚡ [TIME-TO-FIRST-AUDIO]: ${firstAudioTime}ms!`);
      console.log(`   (Playback starts in player now)\n`);
    }
  }
});

ws.on('close', () => {
  console.log(`Stream closed. Received ${audioChunks.length} audio chunks.`);
  if (audioChunks.length > 0) {
    fs.writeFileSync(OUTPUT_FILE, Buffer.concat(audioChunks));
    console.log(`Saved full audio to ${OUTPUT_FILE}`);
  }
});
