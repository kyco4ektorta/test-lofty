"""
Pydantic models for request/response schemas.
"""

from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class BackendChoice(str, Enum):
    local = "local"
    runpod = "runpod"
    auto = "auto"


class OutputFormat(str, Enum):
    wav = "wav"
    mp3 = "mp3"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=3,
        max_length=512,
        description="Text description of the music to generate",
        example="upbeat electronic dance music with heavy bass and synthesizers, 120 BPM",
    )
    duration: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Duration of the generated track in seconds (1–30)",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Sampling temperature. Higher = more creative/random",
    )
    top_k: int = Field(
        default=250,
        ge=0,
        le=2048,
        description="Top-K sampling. 0 = disabled",
    )
    top_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Nucleus (top-p) sampling. 0 = disabled",
    )
    cfg_coef: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Classifier-free guidance coefficient. Higher = more prompt-adherent",
    )
    backend: BackendChoice = Field(
        default=BackendChoice.auto,
        description="Generation backend: local CPU/GPU, RunPod GPU farm, or auto-select",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.wav,
        description="Output audio format",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation. None = random",
    )
    model_size: Optional[str] = Field(
        default=None,
        description="Override model size: small | medium | large | melody",
    )

    @validator("prompt")
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt must not be empty or whitespace")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "prompt": "cinematic orchestral music with epic strings and drums, emotional, film score",
                "duration": 15,
                "temperature": 1.0,
                "cfg_coef": 3.5,
                "backend": "auto",
                "output_format": "wav",
            }
        }


class GenerationResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier for status polling")
    status: JobStatus
    message: str
    estimated_seconds: int = Field(..., description="Estimated generation time in seconds")

    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "message": "Job queued. Poll /jobs/{job_id} for status.",
                "estimated_seconds": 30,
            }
        }


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    prompt: Optional[str] = None
    duration: Optional[int] = None
    audio_url: Optional[str] = Field(
        None,
        description="Direct URL to generated audio file (available when status=completed)",
    )
    backend_used: Optional[str] = Field(None, description="'local' or 'runpod'")
    error: Optional[str] = Field(None, description="Error message if status=failed")
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = Field(
        None, description="Actual generation wall-clock time"
    )

    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "prompt": "upbeat jazz with piano",
                "duration": 10,
                "audio_url": "https://your-api.replit.app/outputs/550e8400.wav",
                "backend_used": "local",
                "created_at": "2024-01-01T12:00:00",
                "completed_at": "2024-01-01T12:00:45",
                "duration_seconds": 45.2,
            }
        }


class HealthResponse(BaseModel):
    status: str
    version: str
    local_model_loaded: bool
    runpod_configured: bool
    active_jobs: int
    timestamp: datetime


class RunpodConfig(BaseModel):
    api_key: str
    endpoint_id: str
    pod_id: Optional[str] = None
