# MusicGen AI REST API

## Overview
FastAPI-based REST service for AI music generation using Meta's MusicGen model.
Supports local generation and GPU farm (RunPod) offloading with auto-scaling.

## Architecture
- **Framework**: FastAPI (Python 3.12)
- **Server**: Uvicorn on port 5000
- **Package layout**: `app/` package containing all modules

## Project Structure
```
app/
├── __init__.py
├── main.py              # FastAPI app, all endpoints
├── models.py            # Pydantic request/response schemas
├── job_store.py         # In-memory async job tracking
├── local_generator.py   # MusicGen + LoRA inference (mock mode when audiocraft not installed)
└── runpod_client.py     # RunPod GPU farm integration
outputs/                 # Generated audio files
finetune_lora.py         # LoRA fine-tuning script
handler.py               # RunPod worker handler
requirements.txt
```

## Running
```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

## Key Endpoints
- `GET /health` - Service health check
- `POST /generate` - Submit music generation job
- `GET /jobs/{job_id}` - Poll job status
- `GET /jobs` - List all jobs
- `GET /audio/{filename}` - Download audio
- `GET /models` - Model info
- `GET /docs` - Swagger UI

## Environment Variables
- `BASE_URL` - Public URL of the API (e.g., https://your-repl.repl.co)
- `MUSICGEN_MODEL_SIZE` - Model size: small | medium | large | melody (default: small)
- `RUNPOD_API_KEY` - RunPod API key (optional)
- `RUNPOD_ENDPOINT_ID` - RunPod serverless endpoint ID (optional)
- `LORA_WEIGHTS_PATH` - Path to LoRA fine-tuned weights (optional)
- `PRELOAD_MODEL` - Set to "true" to preload model on startup (default: false)

## Notes
- Without audiocraft installed, the API runs in MOCK mode (returns silent WAV files)
- audiocraft requires ~2GB disk and significant RAM; use MUSICGEN_MODEL_SIZE=small on free tier
- RunPod integration enables GPU-accelerated generation for longer tracks
