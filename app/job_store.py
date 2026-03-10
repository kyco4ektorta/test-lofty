"""
In-memory job store. Replace with Redis for production multi-worker setups.
"""

import threading
from datetime import datetime
from typing import Dict, Optional, List
from app.models import JobStatus, GenerationRequest, JobStatusResponse


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, request: GenerationRequest):
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": JobStatus.QUEUED,
                "prompt": request.prompt,
                "duration": request.duration,
                "audio_url": None,
                "backend_used": None,
                "error": None,
                "created_at": datetime.utcnow(),
                "started_at": None,
                "completed_at": None,
            }

    def update(self, job_id: str, **kwargs):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)
                # Compute wall-clock duration
                job = self._jobs[job_id]
                if job.get("started_at") and job.get("completed_at"):
                    delta = job["completed_at"] - job["started_at"]
                    self._jobs[job_id]["duration_seconds"] = round(delta.total_seconds(), 2)

    def get(self, job_id: str) -> Optional[JobStatusResponse]:
        with self._lock:
            data = self._jobs.get(job_id)
            if not data:
                return None
            return JobStatusResponse(**data)

    def delete(self, job_id: str):
        with self._lock:
            self._jobs.pop(job_id, None)

    def count_active(self) -> int:
        with self._lock:
            return sum(
                1 for j in self._jobs.values()
                if j["status"] in (JobStatus.QUEUED, JobStatus.PROCESSING)
            )

    def list_all(self, limit: int = 20, status_filter: Optional[str] = None) -> List[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
            if status_filter:
                jobs = [j for j in jobs if j["status"] == status_filter]
            # Sort newest first
            jobs.sort(key=lambda j: j["created_at"], reverse=True)
            return [JobStatusResponse(**j).dict() for j in jobs[:limit]]


# Singleton
job_store = JobStore()
