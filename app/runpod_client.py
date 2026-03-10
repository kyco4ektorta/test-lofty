"""
RunPod GPU Farm Client
======================
Handles auto-provisioning, job dispatch, and cost management for RunPod GPU instances.

Flow:
  1. Check if endpoint is idle → wake it up
  2. Submit generation job to RunPod serverless endpoint
  3. Poll for result with exponential backoff
  4. Return audio URL (RunPod stores output in its own CDN or S3)
"""

import os
import asyncio
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.io/v2"
RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"


class RunpodClient:
    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def is_configured(self) -> bool:
        return bool(self.api_key and self.endpoint_id)

    # ── Serverless Endpoint (recommended) ────────────────────────────────────

    async def generate(self, job_id: str, request) -> str:
        """
        Submit generation to RunPod serverless endpoint.
        Auto-wakes the endpoint if cold, polls until complete.
        """
        async with httpx.AsyncClient(timeout=300) as client:
            # Submit job
            payload = {
                "input": {
                    "job_id": job_id,
                    "prompt": request.prompt,
                    "duration": request.duration,
                    "temperature": request.temperature,
                    "top_k": request.top_k,
                    "cfg_coef": request.cfg_coef,
                    "output_format": request.output_format,
                    "seed": request.seed,
                    "model_size": request.model_size or "large",
                }
            }

            logger.info(f"[{job_id}] Submitting to RunPod endpoint: {self.endpoint_id}")
            resp = await client.post(
                f"{RUNPOD_API_BASE}/{self.endpoint_id}/run",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            runpod_job = resp.json()
            runpod_job_id = runpod_job["id"]
            logger.info(f"[{job_id}] RunPod job ID: {runpod_job_id}")

            # Poll for completion
            audio_url = await self._poll_job(client, job_id, runpod_job_id)
            return audio_url

    async def _poll_job(
        self, client: httpx.AsyncClient, local_job_id: str, runpod_job_id: str
    ) -> str:
        """Poll RunPod job with exponential backoff until complete."""
        delay = 2
        max_delay = 30
        timeout_at = time.time() + 600  # 10 minute hard timeout

        while time.time() < timeout_at:
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, max_delay)

            resp = await client.get(
                f"{RUNPOD_API_BASE}/{self.endpoint_id}/status/{runpod_job_id}",
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "UNKNOWN")

            logger.info(f"[{local_job_id}] RunPod status: {status}")

            if status == "COMPLETED":
                output = data.get("output", {})
                audio_url = output.get("audio_url") or output.get("url")
                if not audio_url:
                    raise RuntimeError(f"RunPod completed but no audio_url in output: {output}")
                return audio_url

            elif status == "FAILED":
                error = data.get("error", "Unknown RunPod error")
                raise RuntimeError(f"RunPod job failed: {error}")

            elif status in ("CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"RunPod job ended with status: {status}")

            # IN_QUEUE, IN_PROGRESS → keep polling

        raise TimeoutError(f"RunPod job {runpod_job_id} timed out after 10 minutes")

    # ── Pod Management (on-demand GPU instances) ──────────────────────────────

    async def start_instance(self) -> dict:
        """
        Start a RunPod GPU pod on-demand via GraphQL API.
        Use this for longer workloads or batch fine-tuning.
        """
        mutation = """
        mutation {
          podFindAndDeployOnDemand(input: {
            cloudType: SECURE
            gpuCount: 1
            volumeInGb: 20
            containerDiskInGb: 10
            minVcpuCount: 4
            minMemoryInGb: 15
            gpuTypeId: "NVIDIA GeForce RTX 4090"
            name: "musicgen-worker"
            imageName: "your-dockerhub/musicgen-worker:latest"
            dockerArgs: ""
            ports: "8000/http"
            volumeMountPath: "/workspace"
            env: [
              { key: "MODEL_SIZE", value: "large" }
              { key: "HF_TOKEN", value: "" }
            ]
          }) {
            id
            imageName
            machineId
            costPerHr
            gpuDisplayName
          }
        }
        """
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                RUNPOD_GRAPHQL,
                headers=self._headers,
                json={"query": mutation},
            )
            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                raise RuntimeError(f"GraphQL error: {data['errors']}")

            pod = data["data"]["podFindAndDeployOnDemand"]
            logger.info(f"Started RunPod pod: {pod['id']} @ ${pod['costPerHr']}/hr")
            return {
                "status": "started",
                "pod_id": pod["id"],
                "gpu": pod["gpuDisplayName"],
                "cost_per_hour": pod["costPerHr"],
            }

    async def stop_instance(self, pod_id: Optional[str] = None) -> dict:
        """Stop a RunPod pod to save costs."""
        pid = pod_id or os.getenv("RUNPOD_POD_ID", "")
        if not pid:
            return {"status": "error", "message": "No pod_id provided or set in RUNPOD_POD_ID"}

        mutation = f"""
        mutation {{
          podStop(input: {{ podId: "{pid}" }}) {{
            id
            desiredStatus
          }}
        }}
        """
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                RUNPOD_GRAPHQL,
                headers=self._headers,
                json={"query": mutation},
            )
            resp.raise_for_status()
            data = resp.json()
            return {"status": "stopped", "pod_id": pid, "response": data}

    async def get_status(self) -> dict:
        """Get current RunPod endpoint and pod status."""
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                resp = await client.get(
                    f"{RUNPOD_API_BASE}/{self.endpoint_id}/health",
                    headers=self._headers,
                )
                health = resp.json() if resp.status_code == 200 else {"error": resp.text}
            except Exception as e:
                health = {"error": str(e)}

        return {
            "configured": True,
            "endpoint_id": self.endpoint_id,
            "health": health,
            "api_key_set": bool(self.api_key),
        }
