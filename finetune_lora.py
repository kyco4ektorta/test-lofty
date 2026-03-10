"""
LoRA Fine-tuning Script for MusicGen
=====================================
Fine-tune MusicGen on a custom music dataset using LoRA adapters.

Requirements:
  - CUDA GPU with 8GB+ VRAM (for musicgen-medium)
  - Dataset: folder with .wav files + matching .txt description files

Usage:
  python finetune_lora.py \
    --model_size medium \
    --data_dir ./my_music_dataset \
    --output_dir ./lora_weights \
    --epochs 10 \
    --lr 3e-4

Dataset format:
  my_music_dataset/
    track_001.wav
    track_001.txt   ← "upbeat jazz piano with drums, 120 BPM"
    track_002.wav
    track_002.txt   ← "dark ambient drone, cinematic, slow"
"""

import os
import argparse
import logging
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── LoRA Layer ────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    LoRA adapter for nn.Linear layers.
    Adds trainable low-rank matrices A and B:
        output = W·x + (B·A·x) * (alpha/rank)

    Only A and B are trained; original W is frozen.
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        d_out, d_in = original.weight.shape

        # Initialize: A ~ N(0, 0.02), B = 0 (so delta starts at zero)
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

        # Freeze original weights
        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

    def forward(self, x):
        base = self.original(x)
        lora_delta = (x @ self.lora_A.T @ self.lora_B.T) * (self.alpha / self.rank)
        return base + lora_delta


def inject_lora(model, rank: int = 8, alpha: float = 16.0, target_modules=None):
    """
    Replace Linear layers in target_modules with LoRA-wrapped versions.
    Default targets: attention q, k, v, out projections.
    """
    if target_modules is None:
        target_modules = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}

    replaced = 0
    for name, module in model.lm.transformer.named_modules():
        if isinstance(module, nn.Linear):
            leaf_name = name.split(".")[-1]
            if leaf_name in target_modules:
                parent_name = ".".join(name.split(".")[:-1])
                parent = model.lm.transformer
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
                setattr(parent, leaf_name, LoRALinear(module, rank=rank, alpha=alpha))
                replaced += 1

    # Freeze all non-LoRA params
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if not param.requires_grad:
            pass
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)

    logger.info(f"LoRA injected: {replaced} layers | "
                f"Trainable params: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)")
    return model


def save_lora_weights(model, output_path: str):
    """Save only LoRA adapter weights (much smaller than full checkpoint)."""
    lora_state = {}
    for name, param in model.lm.transformer.named_parameters():
        if "lora_A" in name or "lora_B" in name or "lora_alpha" in name:
            lora_state[name] = param.data.cpu()

    torch.save(lora_state, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    logger.info(f"LoRA weights saved: {output_path} ({size_mb:.1f} MB, {len(lora_state)} tensors)")


# ── Dataset ───────────────────────────────────────────────────────────────────

class MusicDataset(Dataset):
    """
    Paired audio + text description dataset for MusicGen fine-tuning.

    Expects:
      data_dir/
        *.wav  (audio clips, ideally 5-30 seconds each)
        *.txt  (one-line text description matching each .wav)
    """

    def __init__(self, data_dir: str, sample_rate: int = 32000, max_duration: float = 20.0):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.pairs = self._load_pairs()
        logger.info(f"Dataset: {len(self.pairs)} audio-text pairs from {data_dir}")

    def _load_pairs(self) -> List[Tuple[Path, str]]:
        pairs = []
        for wav_path in sorted(self.data_dir.glob("*.wav")):
            txt_path = wav_path.with_suffix(".txt")
            if txt_path.exists():
                description = txt_path.read_text(encoding="utf-8").strip()
                if description:
                    pairs.append((wav_path, description))
            else:
                logger.warning(f"No .txt for {wav_path.name}, skipping")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        wav_path, description = self.pairs[idx]
        try:
            import torchaudio
            wav, sr = torchaudio.load(str(wav_path))
            # Resample if needed
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            # Mix to mono
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            # Pad or truncate
            if wav.shape[1] > self.max_samples:
                wav = wav[:, :self.max_samples]
            else:
                wav = torch.nn.functional.pad(wav, (0, self.max_samples - wav.shape[1]))
            return {"wav": wav.squeeze(0), "description": description}
        except Exception as e:
            logger.error(f"Error loading {wav_path}: {e}")
            return {"wav": torch.zeros(self.max_samples), "description": description}


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    from audiocraft.models import MusicGen

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device}")

    # Load model + inject LoRA
    logger.info(f"Loading MusicGen {args.model_size}...")
    model = MusicGen.get_pretrained(f"facebook/musicgen-{args.model_size}")
    model = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model.to(device)

    dataset = MusicDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(loader):
            wavs = batch["wav"].to(device)          # [B, T]
            descriptions = batch["description"]      # List[str]

            try:
                # Encode audio with EnCodec
                with torch.no_grad():
                    audio_tokens, _ = model.compression_model.encode(
                        wavs.unsqueeze(1)  # [B, 1, T]
                    )

                # Get text conditioning tokens
                attributes, _ = model._prepare_tokens_and_attributes(descriptions, None)

                # Compute LM loss (next-token prediction over audio tokens)
                loss = model.lm.compute_predictions(
                    audio_tokens, [], attributes
                ).loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

                if step % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}/{args.epochs} | Step {step}/{len(loader)} "
                        f"| Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

            except Exception as e:
                logger.error(f"Training step failed: {e}")
                continue

        avg_loss = epoch_loss / max(len(loader), 1)
        history.append({"epoch": epoch, "loss": avg_loss})
        logger.info(f"Epoch {epoch} complete | Avg loss: {avg_loss:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, "lora_best.pt")
            save_lora_weights(model, ckpt_path)
            logger.info(f"New best checkpoint saved (loss={best_loss:.4f})")

        # Save every N epochs
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"lora_epoch_{epoch}.pt")
            save_lora_weights(model, ckpt_path)

    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete! Best loss: {best_loss:.4f}")
    logger.info(f"Weights saved to: {args.output_dir}/lora_best.pt")
    logger.info("Set LORA_WEIGHTS_PATH=./lora_weights/lora_best.pt to use in API")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MusicGen with LoRA")
    parser.add_argument("--model_size", default="medium",
                        choices=["small", "medium", "large", "melody"],
                        help="MusicGen model size")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with .wav + .txt pairs")
    parser.add_argument("--output_dir", default="./lora_weights",
                        help="Where to save LoRA adapter weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank (higher = more capacity, more VRAM)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA scaling factor")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        parser.error(f"Data directory not found: {args.data_dir}")

    logger.info(f"Starting LoRA fine-tuning: model={args.model_size}, "
                f"rank={args.lora_rank}, alpha={args.lora_alpha}, "
                f"lr={args.lr}, epochs={args.epochs}")
    train(args)


if __name__ == "__main__":
    main()
