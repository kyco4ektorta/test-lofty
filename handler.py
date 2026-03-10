"""
RunPod Serverless Worker — runs ON the GPU instance
====================================================
Deploy this as a Docker container to RunPod.
It receives jobs from the RunPod queue and runs MusicGen generation.

Build and push:
  docker build -t youruser/musicgen-worker:latest .
  docker push youruser/musicgen-worker:latest
"""

import os
import time
import uuid
import base64
import logging
import tempfile
from typing import Any

import runpod  # pip install runpod

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Model loading (once at cold start) ───────────────────────────────────────
_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    from audiocraft.models import MusicGen

    model_size = os.getenv("MODEL_SIZE", "large")
    logger.info(f"Cold start: loading MusicGen {model_size}...")
    t0 = time.time()
    _model = MusicGen.get_pretrained(f"facebook/musicgen-{model_size}")
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Apply LoRA weights if mounted
    lora_path = os.getenv("LORA_WEIGHTS_PATH", "/workspace/lora_weights.pt")
    if os.path.exists(lora_path):
        logger.info(f"Loading LoRA weights from {lora_path}")
        _apply_lora(_model, lora_path)

    return _model


def _apply_lora(model, lora_path: str):
    import torch
    state_dict = torch.load(lora_path, map_location="cpu")
    transformer = model.lm.transformer
    loaded = 0
    for name, param in transformer.named_parameters():
        if f"{name}.lora_A" in state_dict and f"{name}.lora_B" in state_dict:
            A = state_dict[f"{name}.lora_A"]
            B = state_dict[f"{name}.lora_B"]
            alpha = state_dict.get(f"{name}.lora_alpha", 1.0)
            rank = A.shape[0]
            param.data += (B @ A) * (alpha / rank)
            loaded += 1
    logger.info(f"LoRA: applied {loaded} adapter layers")


# ── RunPod handler ────────────────────────────────────────────────────────────

def handler(job: dict) -> dict[str, Any]:
    """
    RunPod job handler.
    Receives job input, runs MusicGen, returns audio as base64 + URL.
    """
    job_input = job.get("input", {})
    job_id = job_input.get("job_id", str(uuid.uuid4()))

    logger.info(f"[{job_id}] Received job: {job_input}")

    try:
        model = load_model()

        prompt = job_input["prompt"]
        duration = int(job_input.get("duration", 10))
        temperature = float(job_input.get("temperature", 1.0))
        top_k = int(job_input.get("top_k", 250))
        cfg_coef = float(job_input.get("cfg_coef", 3.0))
        output_format = job_input.get("output_format", "wav")
        seed = job_input.get("seed")

        import torch
        if seed is not None:
            torch.manual_seed(int(seed))

        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            cfg_coef=cfg_coef,
        )

        logger.info(f"[{job_id}] Generating: '{prompt[:60]}'")
        t0 = time.time()
        wav = model.generate([prompt])
        elapsed = time.time() - t0
        logger.info(f"[{job_id}] Done in {elapsed:.1f}s")

        # Save to temp file
        from audiocraft.data.audio import audio_write
        with tempfile.TemporaryDirectory() as tmpdir:
            stem = f"{tmpdir}/{job_id}"
            audio_write(stem, wav[0].cpu(), model.sample_rate, strategy="loudness")
            wav_path = f"{stem}.wav"

            # Upload to S3/R2 if configured
            audio_url = _upload_audio(wav_path, job_id, output_format)

            # Also return base64 as fallback
            with open(wav_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()

        return {
            "job_id": job_id,
            "audio_url": audio_url,
            "audio_base64": audio_b64,
            "format": output_format,
            "duration_seconds": duration,
            "generation_time_seconds": round(elapsed, 2),
            "model_size": os.getenv("MODEL_SIZE", "large"),
        }

    except Exception as e:
        logger.error(f"[{job_id}] Handler error: {e}", exc_info=True)
        return {"error": str(e), "job_id": job_id}


def _upload_audio(wav_path: str, job_id: str, output_format: str) -> str:
    """Upload audio to S3/R2 bucket. Returns public URL."""
    bucket = os.getenv("S3_BUCKET", "")
    if not bucket:
        # Return base64 data URI as fallback
        with open(wav_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:audio/wav;base64,{data[:50]}..."  # Truncated in URL field

    try:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        key = f"musicgen/{job_id}.{output_format}"
        s3.upload_file(
            wav_path, bucket, key,
            ExtraArgs={"ContentType": f"audio/{output_format}"}
        )
        region = os.getenv("AWS_REGION", "us-east-1")
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return ""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
