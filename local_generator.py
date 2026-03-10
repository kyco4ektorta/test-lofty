"""
Local MusicGen generator using Meta's audiocraft library.
Supports LoRA fine-tuned weights for enhanced quality.
"""

import os
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded to avoid startup crashes when audiocraft isn't installed
_musicgen = None
_model = None
_executor = ThreadPoolExecutor(max_workers=2)


def _load_model(model_size: str, lora_path: str):
    """Load MusicGen model synchronously (runs in thread pool)."""
    global _musicgen, _model

    try:
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        _musicgen = {"MusicGen": MusicGen, "audio_write": audio_write}

        logger.info(f"Loading MusicGen model: facebook/musicgen-{model_size}")
        model = MusicGen.get_pretrained(f"facebook/musicgen-{model_size}")
        model.set_generation_params(duration=10)

        # Apply LoRA weights if available
        if lora_path and os.path.exists(lora_path):
            logger.info(f"Applying LoRA weights from: {lora_path}")
            _apply_lora(model, lora_path)

        _model = model
        logger.info("Model loaded successfully")
        return model

    except ImportError:
        logger.warning(
            "audiocraft not installed — running in MOCK mode. "
            "Install with: pip install audiocraft"
        )
        return None


def _apply_lora(model, lora_path: str):
    """
    Apply LoRA fine-tuned adapter weights to MusicGen transformer.

    LoRA (Low-Rank Adaptation) modifies attention layers with low-rank
    matrices A and B, where the weight update is: ΔW = B·A (rank << d_model).
    This allows fine-tuning on ~0.1% of parameters while achieving 90%+ of
    full fine-tune quality.
    """
    try:
        import torch
        state_dict = torch.load(lora_path, map_location="cpu")

        # MusicGen's transformer is in model.lm.transformer
        transformer = model.lm.transformer

        loaded, skipped = 0, 0
        for name, param in transformer.named_parameters():
            lora_key_a = f"{name}.lora_A"
            lora_key_b = f"{name}.lora_B"

            if lora_key_a in state_dict and lora_key_b in state_dict:
                # Apply LoRA delta: W_new = W_original + B @ A * (alpha/rank)
                lora_A = state_dict[lora_key_a]
                lora_B = state_dict[lora_key_b]
                alpha = state_dict.get(f"{name}.lora_alpha", 1.0)
                rank = lora_A.shape[0]
                delta = (lora_B @ lora_A) * (alpha / rank)
                param.data += delta.to(param.device)
                loaded += 1
            else:
                skipped += 1

        logger.info(f"LoRA applied: {loaded} layers updated, {skipped} layers skipped")

    except Exception as e:
        logger.error(f"LoRA application failed: {e}. Continuing without LoRA.")


def _generate_sync(job_id: str, prompt: str, duration: int, temperature: float,
                   top_k: int, top_p: float, cfg_coef: float,
                   output_format: str, seed: Optional[int]) -> str:
    """Synchronous generation — runs in thread pool."""

    # ── MOCK MODE (no audiocraft installed) ──────────────────────────────────
    if _model is None or _musicgen is None:
        logger.info(f"[{job_id}] MOCK MODE: simulating generation ({duration}s)")
        # Create a minimal valid WAV file (44 bytes header + silence)
        time.sleep(min(duration * 0.5, 5))  # Simulate processing time
        output_path = f"outputs/{job_id}.wav"
        os.makedirs("outputs", exist_ok=True)
        _write_mock_wav(output_path, duration)
        logger.info(f"[{job_id}] MOCK: written to {output_path}")
        return output_path

    # ── REAL GENERATION ──────────────────────────────────────────────────────
    import torch
    audio_write = _musicgen["audio_write"]

    if seed is not None:
        torch.manual_seed(seed)

    _model.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p if top_p > 0 else None,
        cfg_coef=cfg_coef,
    )

    logger.info(f"[{job_id}] Generating: '{prompt[:60]}...' | duration={duration}s")
    t0 = time.time()

    wav = _model.generate([prompt])  # shape: [1, C, T]

    elapsed = time.time() - t0
    logger.info(f"[{job_id}] Generated in {elapsed:.1f}s")

    os.makedirs("outputs", exist_ok=True)
    stem = f"outputs/{job_id}"
    audio_write(stem, wav[0].cpu(), _model.sample_rate, strategy="loudness")

    # audio_write appends .wav automatically
    wav_path = f"{stem}.wav"

    # Convert to MP3 if requested
    if output_format == "mp3":
        mp3_path = f"{stem}.mp3"
        _to_mp3(wav_path, mp3_path)
        return mp3_path

    return wav_path


def _write_mock_wav(path: str, duration_seconds: int):
    """Write a silent but valid WAV file for mock mode."""
    import struct, math
    sample_rate = 22050
    num_samples = sample_rate * duration_seconds
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate,
                            byte_rate, block_align, bits_per_sample))
        # data chunk (silence)
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * int(data_size))


def _to_mp3(wav_path: str, mp3_path: str):
    """Convert WAV to MP3 using ffmpeg or pydub."""
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-q:a", "2", mp3_path],
            check=True, capture_output=True
        )
        os.remove(wav_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ffmpeg not available, keeping WAV format")


class LocalGenerator:
    def __init__(self, model_size: str = "small", lora_weights_path: str = ""):
        self.model_size = model_size
        self.lora_weights_path = lora_weights_path
        self._loading = False

        # Pre-load model in background (non-blocking startup)
        if os.getenv("PRELOAD_MODEL", "false").lower() == "true":
            asyncio.get_event_loop().run_in_executor(
                _executor, _load_model, model_size, lora_weights_path
            )

    def is_loaded(self) -> bool:
        return _model is not None

    async def generate(self, job_id: str, request) -> str:
        """Async generation wrapper — offloads to thread pool."""
        global _model, _musicgen

        # Load model on first use
        if _model is None and not self._loading:
            self._loading = True
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _executor, _load_model,
                request.model_size or self.model_size,
                self.lora_weights_path
            )
            self._loading = False

        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            _executor,
            _generate_sync,
            job_id,
            request.prompt,
            request.duration,
            request.temperature,
            request.top_k,
            request.top_p,
            request.cfg_coef,
            request.output_format,
            request.seed,
        )
        return output_path
