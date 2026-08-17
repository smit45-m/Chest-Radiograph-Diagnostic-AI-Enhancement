"""End-to-end diagnosis pipeline: image → classify → heatmap → report.

This is the single entry point consumed by both the FastAPI server and
the CLI ``scripts/predict.py``.  It owns the model, Grad-CAM context,
and the report generator so callers never touch PyTorch directly.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cxr.constants import PATHOLOGIES
from cxr.data.labels import decode_findings
from cxr.data.transforms import infer_transform
from cxr.interpret.gradcam import GradCAM, denormalize, overlay_to_png_bytes
from cxr.models.densenet import DenseNet121Classifier
from cxr.reports.generator import generate_report


class DiagnosisPipeline:
    """Load a trained CXR model and run the full predict → heatmap → report pipeline."""

    def __init__(
        self,
        checkpoint_path: str | Path = "checkpoints/best.pt",
        config: Any | None = None,
        device: str | None = None,
        threshold: float = 0.45,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold = threshold
        self._config = config

        # Image size from config or default.
        self.image_size: int = 224
        if config is not None:
            self.image_size = int(config.get("data.image_size", 224))

        # Load model.
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            self.model = DenseNet121Classifier.load(ckpt, map_location=self.device)
            print(f"[pipeline] Loaded checkpoint from {ckpt}")
        else:
            self.model = DenseNet121Classifier(pretrained="imagenet")
            print(f"[pipeline] No checkpoint at {ckpt}; using ImageNet init")
        self.model = self.model.to(self.device).eval()

        # Grad-CAM targeting the last dense block.
        self._gradcam = GradCAM(self.model, self.model.cam_target_layer())

        # Preprocessing transform (eval mode).
        self._transform = infer_transform(self.image_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_input: bytes | Image.Image, threshold: float | None = None) -> dict:
        """Run full pipeline on a single image.

        Parameters
        ----------
        image_input : bytes or PIL.Image.Image
            Raw JPEG/PNG bytes *or* an already-opened PIL image.
        threshold : float, optional
            Override the instance threshold for this call.

        Returns
        -------
        dict with keys:
            predictions  – list[dict] with name, score, severity for all 14
            positives    – list[dict] only above-threshold findings
            heatmaps     – list[dict] with pathology + base64 PNG overlay
            report       – dict from generate_report()
            inference_time_ms – float
        """
        t0 = time.perf_counter()
        thr = threshold if threshold is not None else self.threshold

        # --- decode image ---
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("L")
        else:
            pil_img = image_input.convert("L")

        # --- preprocess ---
        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)

        # --- forward pass ---
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # --- build scores dict ---
        scores: dict[str, float] = {
            name: round(float(probs[i]), 4) for i, name in enumerate(PATHOLOGIES)
        }

        # --- severity helper ---
        def _severity(s: float) -> str:
            if s >= 0.85:
                return "severe"
            if s >= 0.65:
                return "moderate"
            return "mild"

        all_preds = [
            {"name": name, "score": scores[name], "severity": _severity(scores[name])}
            for name in PATHOLOGIES
        ]
        positives = [p for p in all_preds if p["score"] >= thr]

        # --- Grad-CAM heatmaps for positive findings ---
        heatmaps: list[dict[str, str]] = []
        if positives:
            # Need gradients for Grad-CAM, so re-enable.
            tensor_grad = self._transform(pil_img).unsqueeze(0).to(self.device)
            tensor_grad.requires_grad_(True)
            rgb = denormalize(tensor_grad)

            for pred in positives:
                idx = PATHOLOGIES.index(pred["name"])
                cam = self._gradcam.generate(tensor_grad, class_idx=idx)
                png_bytes = overlay_to_png_bytes(rgb, cam, alpha=0.45)
                heatmaps.append({
                    "pathology": pred["name"],
                    "image_base64": base64.b64encode(png_bytes).decode("ascii"),
                })

        # --- report generation ---
        report = generate_report(
            scores=scores,
            threshold=thr,
            image_id=None,
            use_t5=True,
            use_llm=True,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "predictions": all_preds,
            "positives": positives,
            "heatmaps": heatmaps,
            "report": report,
            "inference_time_ms": round(elapsed_ms, 2),
        }

    def forward_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Raw batched forward pass for :class:`DynamicBatcher`.

        Parameters
        ----------
        batch : Tensor of shape (B, 3, H, W)

        Returns
        -------
        Tensor of shape (B, 14) with sigmoid probabilities.
        """
        batch = batch.to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
        return torch.sigmoid(logits).cpu()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release Grad-CAM hooks."""
        self._gradcam.close()

    def __enter__(self) -> "DiagnosisPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
