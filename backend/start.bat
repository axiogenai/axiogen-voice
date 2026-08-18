@echo off
echo === Starting Axiogen Voice Standalone Server ===
docker compose down
docker compose build
docker compose up -d
echo.
echo === Axiogen Voice Server Running on http://localhost:7860 ===
pause
