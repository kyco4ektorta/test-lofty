"""
MusicGen AI REST API
====================
FastAPI-based REST service for AI music generation using Meta's MusicGen model.
Supports local generation and GPU farm (RunPod) offloading with auto-scaling.
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))

from app.models import (
    GenerationRequest, GenerationResponse, JobStatus,
    JobStatusResponse, HealthResponse, RunpodConfig
)
from app.job_store import job_store
from app.runpod_client import RunpodClient
from app.local_generator import LocalGenerator

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="🎵 MusicGen AI API",
    description="""
## AI Music Generation Service

Generate high-quality music from text descriptions using Meta's MusicGen model
with optional GPU farm acceleration via RunPod.

### Features
- 🎵 Text-to-music generation (up to 30 seconds)
- 🚀 Auto-scaling GPU farm integration (RunPod)
- 🎛️ Fine-tuned model support via LoRA weights
- 📊 Async job queue with status polling
- 🎼 Multiple output formats (wav, mp3)

### Quick Start
```bash
curl -X POST /generate \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "upbeat electronic dance music with heavy bass", "duration": 10}'
```
    """,
    version="1.0.0",
    contact={"name": "Music AI API", "url": "https://github.com/yourrepo/musicgen-api"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated audio files
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ─── Clients ─────────────────────────────────────────────────────────────────
runpod_client = RunpodClient(
    api_key=os.getenv("RUNPOD_API_KEY", ""),
    endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID", ""),
)
local_generator = LocalGenerator(
    model_size=os.getenv("MUSICGEN_MODEL_SIZE", "small"),
    lora_weights_path=os.getenv("LORA_WEIGHTS_PATH", ""),
)


# ─── Background task ─────────────────────────────────────────────────────────
async def process_generation(job_id: str, request: GenerationRequest):
    """Async worker: runs generation locally or on RunPod GPU farm."""
    try:
        job_store.update(job_id, status=JobStatus.PROCESSING, started_at=datetime.utcnow())
        logger.info(f"[{job_id}] Starting generation | backend={request.backend}")

        use_runpod = (
            request.backend == "runpod"
            or (request.backend == "auto" and request.duration > 15)
        )

        if use_runpod and runpod_client.is_configured():
            logger.info(f"[{job_id}] Dispatching to RunPod GPU farm")
            audio_url = await runpod_client.generate(job_id, request)
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                audio_url=audio_url,
                backend_used="runpod",
                completed_at=datetime.utcnow(),
            )
        else:
            logger.info(f"[{job_id}] Generating locally with MusicGen")
            output_path = await local_generator.generate(job_id, request)
            base_url = os.getenv("BASE_URL", "https://test-lofty--warabooot.replit.app")
            audio_url = f"{base_url}/outputs/{os.path.basename(output_path)}"
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                audio_url=audio_url,
                backend_used="local",
                completed_at=datetime.utcnow(),
            )

        logger.info(f"[{job_id}] Generation complete → {audio_url}")

    except Exception as e:
        logger.error(f"[{job_id}] Generation failed: {e}", exc_info=True)
        job_store.update(job_id, status=JobStatus.FAILED, error=str(e))


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service health check — returns component status."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        local_model_loaded=local_generator.is_loaded(),
        runpod_configured=runpod_client.is_configured(),
        active_jobs=job_store.count_active(),
        timestamp=datetime.utcnow(),
    )


@app.post(
    "/generate",
    response_model=GenerationResponse,
    status_code=202,
    tags=["Music Generation"],
    summary="Submit music generation job",
)
async def generate_music(request: GenerationRequest, background_tasks: BackgroundTasks):
    """
    Submit an async music generation job.

    Returns a **job_id** immediately. Poll `/jobs/{job_id}` for status.

    ### Parameters
    - **prompt**: Text description of the music (e.g. *"cinematic orchestral with piano"*)
    - **duration**: Track length in seconds (1–30)
    - **temperature**: Sampling temperature — higher = more creative (0.1–2.0)
    - **top_k**: Top-K sampling (0 = disabled)
    - **cfg_coef**: Classifier-free guidance scale — higher = more prompt-adherent
    - **backend**: `local` | `runpod` | `auto` (auto uses RunPod for duration > 15s)
    - **output_format**: `wav` or `mp3`
    - **seed**: Optional random seed for reproducibility
    """
    job_id = str(uuid.uuid4())
    job_store.create(job_id, request)
    background_tasks.add_task(process_generation, job_id, request)

    return GenerationResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message="Job queued. Poll /jobs/{job_id} for status.",
        estimated_seconds=request.duration * 3,
    )


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Music Generation"],
    summary="Get job status and result",
)
async def get_job_status(job_id: str):
    """
    Poll this endpoint after submitting a generation job.

    When `status` is **completed**, the `audio_url` field contains a direct link
    to the generated audio file.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get(
    "/jobs",
    tags=["Music Generation"],
    summary="List all jobs",
)
async def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status"),
):
    """List recent generation jobs with optional status filter."""
    jobs = job_store.list_all(limit=limit, status_filter=status)
    return {"jobs": jobs, "total": len(jobs)}


@app.delete("/jobs/{job_id}", tags=["Music Generation"], summary="Cancel/delete a job")
async def delete_job(job_id: str):
    """Cancel a queued job or remove a completed job record."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    job_store.delete(job_id)
    return {"message": f"Job '{job_id}' removed"}


@app.get(
    "/audio/{filename}",
    tags=["Audio Files"],
    summary="Download generated audio",
)
async def download_audio(filename: str):
    """Download a generated audio file by filename."""
    path = f"outputs/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get(
    "/models",
    tags=["System"],
    summary="Available model info",
)
async def get_models():
    """Returns information about available MusicGen model variants."""
    return {
        "models": [
            {
                "id": "musicgen-small",
                "params": "300M",
                "vram_gb": 2,
                "quality": "good",
                "speed": "fast",
                "recommended_for": "prototyping, short clips",
            },
            {
                "id": "musicgen-medium",
                "params": "1.5B",
                "vram_gb": 6,
                "quality": "better",
                "speed": "medium",
                "recommended_for": "general use",
            },
            {
                "id": "musicgen-large",
                "params": "3.3B",
                "vram_gb": 12,
                "quality": "best",
                "speed": "slow",
                "recommended_for": "production quality, GPU farm",
            },
            {
                "id": "musicgen-melody",
                "params": "1.5B",
                "vram_gb": 6,
                "quality": "better",
                "speed": "medium",
                "recommended_for": "melody conditioning",
            },
        ],
        "current_model": os.getenv("MUSICGEN_MODEL_SIZE", "small"),
        "lora_active": bool(os.getenv("LORA_WEIGHTS_PATH", "")),
    }


@app.post(
    "/runpod/start",
    tags=["GPU Farm"],
    summary="Manually start RunPod GPU instance",
)
async def start_runpod():
    """Manually spin up a RunPod GPU instance for faster generation."""
    if not runpod_client.is_configured():
        raise HTTPException(status_code=400, detail="RunPod not configured (set RUNPOD_API_KEY)")
    result = await runpod_client.start_instance()
    return result


@app.post(
    "/runpod/stop",
    tags=["GPU Farm"],
    summary="Manually stop RunPod GPU instance",
)
async def stop_runpod():
    """Stop the RunPod GPU instance to save costs."""
    if not runpod_client.is_configured():
        raise HTTPException(status_code=400, detail="RunPod not configured")
    result = await runpod_client.stop_instance()
    return result


@app.get(
    "/runpod/status",
    tags=["GPU Farm"],
    summary="Get RunPod instance status",
)
async def runpod_status():
    """Check current RunPod GPU instance status and cost metrics."""
    if not runpod_client.is_configured():
        return {"configured": False, "message": "Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID env vars"}
    return await runpod_client.get_status()


@app.get("/", include_in_schema=False)
async def serve_ui():
    from fastapi.responses import FileResponse
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return FileResponse(os.path.join(base, "index.html"))
