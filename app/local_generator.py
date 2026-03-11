import os
import asyncio
import logging
import time
import struct
import urllib.request
import urllib.parse
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)


def _write_mock_wav(path: str, duration_seconds: int):
    sample_rate = 22050
    num_samples = sample_rate * duration_seconds
    byte_rate = sample_rate * 2
    data_size = num_samples * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * int(data_size))


def _generate_sync(job_id: str, prompt: str, duration: int,
                   temperature: float, top_k: int, top_p: float,
                   cfg_coef: float, output_format: str,
                   seed: Optional[int]) -> str:

    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/{job_id}.wav"

    colab_url = os.getenv("COLAB_API_URL", "")

    # ── COLAB MODE ────────────────────────────────────────────────────────
    if colab_url:
        try:
            encoded_prompt = urllib.parse.quote(prompt)
            duration_tokens = duration * 50
            url = f"{colab_url}/generate?prompt={encoded_prompt}&duration_tokens={duration_tokens}"
            logger.info(f"[{job_id}] Sending POST to Colab: {url}")

            # POST запрос с пустым телом
            req = urllib.request.Request(url, data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=300) as response:
                audio_data = response.read()

            with open(output_path, "wb") as f:
                f.write(audio_data)

            logger.info(f"[{job_id}] Colab generation complete → {output_path} ({len(audio_data)} bytes)")
            return output_path

        except Exception as e:
            logger.error(f"[{job_id}] Colab failed: {e}. Falling back to mock.")

    # ── MOCK MODE ─────────────────────────────────────────────────────────
    logger.info(f"[{job_id}] MOCK MODE: generating silent WAV ({duration}s)")
    time.sleep(2)
    _write_mock_wav(output_path, duration)
    return output_path


class LocalGenerator:
    def __init__(self, model_size: str = "small", lora_weights_path: str = ""):
        self.model_size = model_size
        self.lora_weights_path = lora_weights_path

    def is_loaded(self) -> bool:
        return bool(os.getenv("COLAB_API_URL", ""))

    async def generate(self, job_id: str, request) -> str:
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
