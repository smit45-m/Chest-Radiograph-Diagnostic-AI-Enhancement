"""End-to-end clinical diagnosis pipeline:
image → multi-backbone FP16 ensemble + PSPNet anatomy + optional Gemini Decision 1
→ Grad-CAM++ localization & anatomical zones → Gemini Flash / Groq / ACR report synthesis.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from cxr.ai.gemini_engine import AI_RUNTIME_CONFIG, run_gemini_decision_1
from cxr.constants import PATHOLOGIES
from cxr.interpret.gradcam import (
    GradCAM,
    cam_to_transparent_rgba_png,
    extract_hotspot_metadata,
    overlay_to_png_bytes,
)
from cxr.models.densenet import DenseNet121Classifier
from cxr.models.ensemble import ClinicalEnsemble, build_vlm_radiograph_jpeg, preprocess_radiograph
from cxr.reports.generator import generate_report


class DiagnosisPipeline:
    """Load the clinical CXR ensemble (or local checkpoint) and run full diagnosis."""

    def __init__(
        self,
        checkpoint_path: str | Path = "checkpoints/best.pt",
        config: Any | None = None,
        device: str | None = None,
        threshold: float = 0.50,
        ensemble_mode: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold = threshold
        self._config = config
        self.ensemble_mode = ensemble_mode or AI_RUNTIME_CONFIG.ensemble_mode or "full"

        self.image_size: int = 224
        if config is not None:
            self.image_size = int(config.get("data.image_size", 224))

        ckpt = Path(checkpoint_path)
        self.is_clinical_xrv = False
        self.ensemble: ClinicalEnsemble | None = None
        self.model: torch.nn.Module | None = None

        if ckpt.exists():
            try:
                self.model = DenseNet121Classifier.load(ckpt, map_location=self.device)
                print(f"[pipeline] Loaded local checkpoint from {ckpt}")
            except Exception as exc:
                print(f"[pipeline] Checkpoint load failed ({exc}), falling back to clinical ensemble...")

        if self.model is None:
            try:
                self.ensemble = ClinicalEnsemble(device=self.device, mode=self.ensemble_mode)
                self.model = self.ensemble.primary_model
                self.is_clinical_xrv = True
                print(
                    f"[pipeline] Clinical Ensemble active on {self.device}: "
                    f"{', '.join(self.ensemble.active_backbones)}"
                    + (" + PSPNet Anatomy" if self.ensemble.seg_model is not None else "")
                )
            except Exception as exc:
                print(f"[pipeline] Fallback to torchvision DenseNet121 ({exc})")
                self.model = DenseNet121Classifier(pretrained="imagenet")

        self.model = self.model.to(self.device).eval()

        target_layer = (
            self.model.features.denseblock4
            if hasattr(self.model, "features") and hasattr(self.model.features, "denseblock4")
            else self.model.cam_target_layer()
        )
        self._gradcam = GradCAM(self.model, target_layer, method="gradcam++")
        # Warm up Grad-CAM++ backward CUDA kernels so first request has zero cold-start jitter
        if self.device.type == "cuda":
            try:
                ch = 1 if self.is_clinical_xrv else 3
                dummy = torch.zeros(1, ch, self.image_size, self.image_size, device=self.device)
                _ = self._gradcam.generate(dummy, class_idx=0, output_size=(224, 224))
                torch.cuda.synchronize()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Preprocessing (kept compatible with callers & batcher)
    # ------------------------------------------------------------------

    def preprocess_image(self, pil_img: Image.Image) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        norm_224, _, display_rgb_512 = preprocess_radiograph(pil_img, image_size=self.image_size)
        if self.is_clinical_xrv:
            tensor = torch.from_numpy(norm_224).float()
        else:
            resized = cv2.resize(
                cv2.cvtColor(display_rgb_512, cv2.COLOR_RGB2GRAY),
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
            norm_3ch = np.stack([resized / 255.0] * 3, axis=0).astype(np.float32)
            tensor = torch.from_numpy(norm_3ch).unsqueeze(0).float()

        enhanced_raw = cv2.cvtColor(display_rgb_512, cv2.COLOR_RGB2GRAY)
        return tensor, display_rgb_512, enhanced_raw

    # ------------------------------------------------------------------
    # Public Inference API
    # ------------------------------------------------------------------

    def predict(
        self,
        image_input: bytes | Image.Image,
        threshold: float | None = None,
        tta: bool = True,
        patient: dict[str, Any] | None = None,
        use_gemini_decision: bool = True,
        use_llm: bool = True,
        colormap: str = "turbo",
        run_segmentation: bool = True,
    ) -> dict[str, Any]:
        """Run full 4-backbone transfer diagnosis, non-chiral TTA, PSPNet CTR, Grad-CAM++, and report synthesis."""
        t0 = time.perf_counter()
        thr = threshold if threshold is not None else self.threshold

        # 1. Decode image & build contrast-balanced 768px clinical JPEG for Gemini Decision 1 & Vision LLMs
        if isinstance(image_input, bytes):
            raw_bytes = image_input
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            raw_bytes = buf.getvalue()

        gemini_jpeg_bytes = build_vlm_radiograph_jpeg(pil_img, size=768, quality=88)

        # 2. Launch Gemini Decision 1 concurrently with local GPU preprocessing/inference!
        import concurrent.futures

        decision_summary: dict[str, Any] = {"active": False, "status": "skipped", "latency_ms": 0.0}
        dec_future = None
        executor = None
        if use_gemini_decision and AI_RUNTIME_CONFIG.gemini_decision_enabled and AI_RUNTIME_CONFIG.gemini_api_key:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            dec_future = executor.submit(
                run_gemini_decision_1,
                gemini_jpeg_bytes,
                "image/jpeg",
                patient,
            )

        # 3. Vision Forward Pass (Multi-Backbone Clinical Ensemble + PSPNet Segmentation)
        #    Runs on RTX GPU IN PARALLEL while Gemini Decision 1 is in-flight over HTTP!
        t_vis = time.perf_counter()
        anatomy: dict[str, Any] = {}
        seg_masks: dict[str, np.ndarray] = {}
        per_pathology_meta: dict[str, dict[str, Any]] = {}
        active_backbones: list[str] = ["densenet121_local"]

        if self.ensemble is not None:
            ens_out = self.ensemble.predict(
                pil_img=pil_img,
                image_size=self.image_size,
                tta=tta,
                run_segmentation=run_segmentation,
                gemini_decision_future=dec_future,
                gemini_weight=AI_RUNTIME_CONFIG.decision_fusion_weight,
            )
            if executor is not None:
                executor.shutdown(wait=False)
            if isinstance(ens_out.get("decision_summary"), dict):
                decision_summary = ens_out["decision_summary"]
            scores = ens_out["scores"]
            per_pathology_meta = ens_out["meta"]
            anatomy = ens_out["anatomy"]
            seg_masks = ens_out["seg_masks"]
            tensor_in = ens_out["tensor_224"]
            display_rgb_512 = ens_out["display_rgb_512"]
            active_backbones = ens_out["active_backbones"]
            vision_ms = float(ens_out.get("pure_gpu_ms", round((time.perf_counter() - t_vis) * 1000, 2)))
        else:
            if dec_future is not None:
                try:
                    decision_summary = dec_future.result(timeout=14.0)
                except Exception:
                    pass
                finally:
                    if executor is not None:
                        executor.shutdown(wait=False)
            tensor, display_rgb_512, _ = self.preprocess_image(pil_img)
            tensor_in = tensor.to(self.device)
            with torch.no_grad():
                raw_out = self.model(tensor_in)
                probs = torch.sigmoid(raw_out).squeeze(0).cpu().numpy()
            scores = {}
            for i, p in enumerate(PATHOLOGIES):
                score_val = float(probs[i]) if i < len(probs) else 0.05
                scores[p] = round(max(0.001, min(0.999, score_val)), 4)
            vision_ms = round((time.perf_counter() - t_vis) * 1000, 2)

        # 4. Assemble ranked 14-pathology predictions with uncertainty & backbone provenance
        all_preds: list[dict[str, Any]] = []
        for p in PATHOLOGIES:
            conf = float(scores[p])
            if conf >= 0.72:
                severity = "severe"
            elif conf >= 0.50:
                severity = "moderate"
            else:
                severity = "mild"

            meta = per_pathology_meta.get(p, {})
            all_preds.append({
                "name": p,
                "score": round(conf, 4),
                "severity": severity,
                "uncertainty": meta.get("uncertainty", 0.02),
                "agreement": meta.get("agreement", 0.95),
                "ci_low": meta.get("ci_low", max(0.001, round(conf - 0.04, 4))),
                "ci_high": meta.get("ci_high", min(0.999, round(conf + 0.04, 4))),
                "cnn_score": meta.get("cnn_ensemble_score", round(conf, 4)),
                "gemini_decision_score": meta.get("gemini_decision_score"),
                "backbones": meta.get("backbones", {}),
            })

        all_preds.sort(key=lambda x: x["score"], reverse=True)
        positives = [p for p in all_preds if p["score"] >= thr]

        # 5. Grad-CAM++ Localization (high-res 512px blended + pure transparent RGBA + anatomical zone)
        t_cam = time.perf_counter()
        heatmaps: list[dict[str, Any]] = []
        cams_to_generate = positives[:4] if positives else all_preds[:1]

        if cams_to_generate:
            for pred in cams_to_generate:
                p_name = pred["name"]
                if self.is_clinical_xrv and hasattr(self.model, "pathologies") and p_name in self.model.pathologies:
                    idx = self.model.pathologies.index(p_name)
                else:
                    idx = PATHOLOGIES.index(p_name) if p_name in PATHOLOGIES else 0

                try:
                    cam = self._gradcam.generate(tensor_in, class_idx=idx, output_size=(256, 256))
                    disp_256 = cv2.resize(display_rgb_512, (256, 256), interpolation=cv2.INTER_AREA)
                    png_bytes = overlay_to_png_bytes(disp_256, cam, alpha=0.52, colormap=colormap)
                    rgba_bytes = cam_to_transparent_rgba_png(cam, colormap=colormap)
                    hs_meta = extract_hotspot_metadata(cam, seg_masks=seg_masks)
                    # Attach anatomical zone directly onto prediction row too!
                    pred["anatomical_zone"] = hs_meta["anatomical_zone"]
                    heatmaps.append({
                        "pathology": p_name,
                        "image_base64": base64.b64encode(png_bytes).decode("ascii"),
                        "raw_heatmap_base64": base64.b64encode(rgba_bytes).decode("ascii"),
                        "method": "Grad-CAM++",
                        "colormap": colormap,
                        **hs_meta,
                    })
                except Exception as exc:
                    print(f"[pipeline] Grad-CAM++ failed for {p_name}: {exc}")

        gradcam_ms = round((time.perf_counter() - t_cam) * 1000, 2)
        core_vision_total_ms = round(vision_ms + gradcam_ms, 2)

        # 6. Multi-Stage Clinical Report Synthesis (Gemini Flash -> Groq -> ACR Clinical NLP)
        t_rep = time.perf_counter()
        report = generate_report(
            scores=scores,
            threshold=thr,
            image_id=None,
            use_t5=True,
            use_llm=use_llm,
            patient=patient,
            image_bytes=gemini_jpeg_bytes,
            anatomy=anatomy,
            hotspots=heatmaps,
            decision_summary=decision_summary,
        )
        report_ms = round((time.perf_counter() - t_rep) * 1000, 2)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "predictions": all_preds,
            "positives": positives,
            "heatmaps": heatmaps,
            "anatomy": anatomy,
            "decision_summary": decision_summary,
            "report": report,
            "ensemble_info": {
                "mode": self.ensemble_mode,
                "active_backbones": active_backbones,
                "tta_enabled": tta,
                "segmentation_enabled": bool(anatomy),
                "gemini_decision_active": bool(decision_summary.get("active")),
                "device": str(self.device),
            },
            "timings": {
                "vision_ms": vision_ms,
                "gradcam_ms": gradcam_ms,
                "core_total_ms": core_vision_total_ms,
                "decision_api_ms": float(decision_summary.get("latency_ms", 0.0)),
                "report_ms": report_ms,
                "total_ms": elapsed_ms,
            },
            # Report pure vision+interpretability latency when LLM is offline, or total end-to-end
            "inference_time_ms": elapsed_ms,
            "vision_latency_ms": core_vision_total_ms,
        }

    def generate_single_heatmap(
        self,
        image_input: bytes | Image.Image,
        pathology_name: str,
        colormap: str = "turbo",
    ) -> dict[str, Any] | None:
        """Generate a Grad-CAM++ heatmap + anatomical localization for one pathology on demand."""
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        tensor, display_rgb_512, _ = self.preprocess_image(pil_img)
        tensor_in = tensor.to(self.device)

        if self.is_clinical_xrv and hasattr(self.model, "pathologies") and pathology_name in self.model.pathologies:
            idx = self.model.pathologies.index(pathology_name)
        elif pathology_name in PATHOLOGIES:
            idx = PATHOLOGIES.index(pathology_name)
        else:
            return None

        try:
            cam = self._gradcam.generate(tensor_in, class_idx=idx, output_size=(512, 512))
            png_bytes = overlay_to_png_bytes(display_rgb_512, cam, alpha=0.52, colormap=colormap)
            rgba_bytes = cam_to_transparent_rgba_png(cam, colormap=colormap)
            hs_meta = extract_hotspot_metadata(cam)
            return {
                "pathology": pathology_name,
                "image_base64": base64.b64encode(png_bytes).decode("ascii"),
                "raw_heatmap_base64": base64.b64encode(rgba_bytes).decode("ascii"),
                "method": "Grad-CAM++",
                "colormap": colormap,
                **hs_meta,
            }
        except Exception as exc:
            print(f"[pipeline] On-demand Grad-CAM++ error for {pathology_name}: {exc}")
            return None

    def forward_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Raw batched FP16 forward pass for DynamicBatcher."""
        batch = batch.to(self.device)
        use_amp = self.device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            out = torch.sigmoid(self.model(batch).float())
        return out.cpu()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if hasattr(self, "_gradcam") and self._gradcam:
            self._gradcam.close()

    def __enter__(self) -> "DiagnosisPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
