#!/bin/bash
# Axiogen Voice - 1-Click Server Startup Script for Linux / VPS
set -e

echo "=== Starting Axiogen Voice Standalone Server ==="
docker compose down || true
docker compose build
docker compose up -d

echo "=== Server is Live on port 7860 ==="
echo "Health check: curl http://localhost:7860/health"
