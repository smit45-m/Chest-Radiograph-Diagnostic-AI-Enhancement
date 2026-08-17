"""Optional local Hugging Face T5 polisher.

Fine-tune with `python scripts/train_report_t5.py`. If no checkpoint is
present we still expose `rewrite_with_t5` so the API can fall back cleanly.
"""

from __future__ import annotations

from pathlib import Path

import torch


DEFAULT_CKPT = Path("checkpoints/report_t5")


def rewrite_with_t5(draft: str, ckpt_dir: str | Path = DEFAULT_CKPT, max_new_tokens: int = 256) -> str:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No local T5 checkpoint at {ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to(device)
    model.eval()
    encoded = tokenizer(
        f"rewrite radiology report: {draft}",
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)
