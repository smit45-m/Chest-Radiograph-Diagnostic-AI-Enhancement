"""End-to-end clinical diagnosis pipeline (v9.0 Velocity & Precision):
image -> two-level SHA-256 LRU cache -> zero-stall 4-backbone FP16 ensemble + PSPNet anatomy
-> single-pass multi-class Grad-CAM++ with soft thoracic masking & pathology-aware localization
-> concurrent Gemini Decision 1 fusion -> projection-aware CTR -> Gemini Flash / Groq ACR synthesis.

Supports both standard synchronous `predict(...)` and progressive streaming `predict_stream(...)`
so clinicians see Vision + Grad-CAM++ + CTR in <400ms while Decision 1 & ACR reports complete.
"""

from __future__ import annotations

import base64
import concurrent.futures
import copy
import hashlib
import io
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from PIL import Image

from cxr.ai.gemini_engine import AI_RUNTIME_CONFIG, run_gemini_decision_1
from cxr.constants import PATHOLOGIES
from cxr.interpret.gradcam import (
    GradCAM,
    apply_soft_thoracic_mask,
    cam_to_transparent_rgba_png,
    extract_hotspot_metadata,
    overlay_to_png_bytes,
)
from cxr.models.densenet import DenseNet121Classifier
from cxr.models.ensemble import ClinicalEnsemble, build_vlm_radiograph_jpeg, preprocess_radiograph
from cxr.reports.generator import generate_report


class _LRUCache:
    """Thread-safe bounded LRU cache."""

    def __init__(self, max_size: int = 32) -> None:
        self._max_size = max(1, max_size)
        self._lock = threading.Lock()
        self._store: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        with self._lock:
            val = self._store.get(key)
            if val is not None:
                self._store.move_to_end(key)
            return val

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class DiagnosisPipeline:
    """Load the clinical CXR ensemble (or local checkpoint) and run full or streaming diagnosis."""

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

        # Shared thread pool for concurrent Gemini Decision 1 HTTP calls (zero per-request pool churn)
        self._io_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="cxr-io"
        )
        # Two-level content-addressed LRU caches:
        # L1 caches CNN+PSPNet features (reused across threshold/colormap/patient changes)
        # L2 caches complete responses (instant <15ms hits on repeat scans/sample switches)
        self._vision_cache = _LRUCache(max_size=24)
        self._result_cache = _LRUCache(max_size=32)

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
                _ = self._gradcam.generate_multi(dummy, class_indices=[0, 1], output_size=(256, 256))
                torch.cuda.synchronize()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Preprocessing & Internal Helpers
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

    @staticmethod
    def _assemble_predictions(
        scores: dict[str, float],
        per_pathology_meta: dict[str, dict[str, Any]],
        thr: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
            unc = float(meta.get("uncertainty", 0.02))
            if conf >= thr:
                review_flag = "positive"
            elif conf >= 0.30:
                review_flag = "borderline"
            elif unc >= 0.16 and conf >= 0.20:
                review_flag = "uncertain"
            else:
                review_flag = "clear"

            all_preds.append({
                "name": p,
                "score": round(conf, 4),
                "severity": severity,
                "review_flag": review_flag,
                "uncertainty": unc,
                "agreement": meta.get("agreement", 0.95),
                "ci_low": meta.get("ci_low", max(0.001, round(conf - 0.04, 4))),
                "ci_high": meta.get("ci_high", min(0.999, round(conf + 0.04, 4))),
                "cnn_score": meta.get("cnn_ensemble_score", round(conf, 4)),
                "gemini_decision_score": meta.get("gemini_decision_score"),
                "backbones": meta.get("backbones", {}),
            })

        all_preds.sort(key=lambda x: x["score"], reverse=True)
        positives = [p for p in all_preds if p["score"] >= thr]
        return all_preds, positives

    def _pathology_to_idx(self, p_name: str) -> int:
        if self.is_clinical_xrv and hasattr(self.model, "pathologies") and p_name in self.model.pathologies:
            return self.model.pathologies.index(p_name)
        return PATHOLOGIES.index(p_name) if p_name in PATHOLOGIES else 0

    def _generate_heatmaps_fast(
        self,
        tensor_in: torch.Tensor,
        display_rgb_512: np.ndarray,
        targets: list[dict[str, Any]],
        seg_masks: dict[str, np.ndarray],
        colormap: str,
        existing_map: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate Grad-CAM++ maps for up to 4 target pathologies in ONE shared forward pass."""
        if not targets:
            return []

        existing = existing_map or {}
        missing = [pred for pred in targets if pred["name"] not in existing]
        generated: dict[str, dict[str, Any]] = dict(existing)

        if missing:
            t_in = tensor_in.to(self.device)
            idx_pairs = [(pred, self._pathology_to_idx(pred["name"])) for pred in missing]
            class_indices = [idx for _, idx in idx_pairs]
            try:
                cams_by_idx = self._gradcam.generate_multi(
                    t_in, class_indices=class_indices, output_size=(256, 256)
                )
            except Exception as exc:
                print(f"[pipeline] Batched Grad-CAM++ fallback: {exc}")
                cams_by_idx = {}
                for _, idx in idx_pairs:
                    try:
                        cams_by_idx[idx] = self._gradcam.generate(t_in, class_idx=idx, output_size=(256, 256))
                    except Exception:
                        pass

            disp_256 = cv2.resize(display_rgb_512, (256, 256), interpolation=cv2.INTER_AREA)
            for pred, idx in idx_pairs:
                p_name = pred["name"]
                raw_cam = cams_by_idx.get(idx)
                if raw_cam is None:
                    continue
                cam = apply_soft_thoracic_mask(raw_cam, seg_masks) if seg_masks else raw_cam
                png_bytes = overlay_to_png_bytes(disp_256, cam, alpha=0.52, colormap=colormap)
                rgba_bytes = cam_to_transparent_rgba_png(cam, colormap=colormap)
                hs_meta = extract_hotspot_metadata(cam, seg_masks=seg_masks, pathology=p_name)
                generated[p_name] = {
                    "pathology": p_name,
                    "image_base64": base64.b64encode(png_bytes).decode("ascii"),
                    "raw_heatmap_base64": base64.b64encode(rgba_bytes).decode("ascii"),
                    "method": "Grad-CAM++",
                    "colormap": colormap,
                    **hs_meta,
                }

        ordered: list[dict[str, Any]] = []
        for pred in targets:
            hm = generated.get(pred["name"])
            if hm is not None:
                pred["anatomical_zone"] = hm["anatomical_zone"]
                ordered.append(hm)
        return ordered

    @staticmethod
    def _build_cache_keys(
        raw_bytes: bytes,
        thr: float,
        tta: bool,
        colormap: str,
        use_gemini_decision: bool,
        use_llm: bool,
        run_segmentation: bool,
        generate_heatmaps: bool,
        patient: dict[str, Any] | None,
        dicom_view: str | None,
    ) -> tuple[str, str]:
        img_hash = hashlib.sha256(raw_bytes).hexdigest()[:24]
        vision_key = f"v1:{img_hash}:tta={int(tta)}:seg={int(run_segmentation)}"
        cfg_sig = (
            f"d={int(use_gemini_decision and AI_RUNTIME_CONFIG.gemini_decision_enabled and bool(AI_RUNTIME_CONFIG.gemini_api_key))}"
            f":l={int(use_llm and AI_RUNTIME_CONFIG.gemini_flash_enabled)}"
            f":m={AI_RUNTIME_CONFIG.gemini_flash_model}"
            f":w={AI_RUNTIME_CONFIG.decision_fusion_weight:.2f}"
        )
        patient_sig = hashlib.sha256(
            json.dumps(patient or {}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12]
        full_key = (
            f"r1:{vision_key}:thr={thr:.3f}:cm={colormap}:hm={int(generate_heatmaps)}"
            f":dv={dicom_view or ''}:p={patient_sig}:{cfg_sig}"
        )
        return vision_key, full_key

    # ------------------------------------------------------------------
    # Progressive Streaming Inference API (NDJSON / SSE Ready)
    # ------------------------------------------------------------------

    def predict_stream(
        self,
        image_input: bytes | Image.Image,
        threshold: float | None = None,
        tta: bool = True,
        patient: dict[str, Any] | None = None,
        use_gemini_decision: bool = True,
        use_llm: bool = True,
        colormap: str = "turbo",
        run_segmentation: bool = True,
        generate_heatmaps: bool = True,
        dicom_view: str | None = None,
        bypass_cache: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield progressive clinical milestones: `vision` (~350ms) -> `consensus` (~1.2s) -> `complete`."""
        t0 = time.perf_counter()
        thr = threshold if threshold is not None else self.threshold

        # 1. Decode image & compute deterministic SHA-256 cache keys
        if isinstance(image_input, bytes):
            raw_bytes = image_input
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=92)
            raw_bytes = buf.getvalue()

        vision_key, full_key = self._build_cache_keys(
            raw_bytes=raw_bytes,
            thr=thr,
            tta=tta,
            colormap=colormap,
            use_gemini_decision=use_gemini_decision,
            use_llm=use_llm,
            run_segmentation=run_segmentation,
            generate_heatmaps=generate_heatmaps,
            patient=patient,
            dicom_view=dicom_view,
        )

        # Level 2 Instant Cache Hit (<15ms repeat clicks / sample switching!)
        if not bypass_cache:
            cached = self._result_cache.get(full_key)
            if cached is not None:
                hit = copy.deepcopy(cached)
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                hit["cache_hit"] = True
                hit["inference_time_ms"] = elapsed_ms
                hit["vision_latency_ms"] = min(elapsed_ms, float(hit.get("vision_latency_ms", elapsed_ms)))
                hit.setdefault("timings", {})["total_ms"] = elapsed_ms
                hit["timings"]["cache_hit"] = 1.0
                yield {"event": "complete", "data": hit}
                return

        gemini_jpeg_bytes = (
            build_vlm_radiograph_jpeg(pil_img, size=768, quality=88)
            if (use_gemini_decision or use_llm)
            else None
        )

        # 2. Launch Gemini Decision 1 concurrently on shared I/O pool while local GPU runs!
        decision_summary: dict[str, Any] = {"active": False, "status": "skipped", "latency_ms": 0.0}
        dec_future: concurrent.futures.Future | None = None
        if (
            use_gemini_decision
            and gemini_jpeg_bytes is not None
            and AI_RUNTIME_CONFIG.gemini_decision_enabled
            and AI_RUNTIME_CONFIG.gemini_api_key
        ):
            dec_future = self._io_pool.submit(
                run_gemini_decision_1,
                gemini_jpeg_bytes,
                "image/jpeg",
                patient,
            )

        # 3. Vision Forward Pass (L1 Vision Cache or Zero-Stall 4-CNN FP16 + PSPNet)
        t_vis = time.perf_counter()
        vision_bundle: dict[str, Any] | None = None if bypass_cache else self._vision_cache.get(vision_key)
        vision_cache_hit = vision_bundle is not None

        if self.ensemble is not None:
            if vision_bundle is None:
                vb = self.ensemble.predict_vision(
                    pil_img=pil_img,
                    image_size=self.image_size,
                    tta=tta,
                    run_segmentation=run_segmentation,
                )
                # Keep CPU copy in L1 cache so GPU VRAM stays constant
                vision_bundle = {
                    **vb,
                    "tensor_224": vb["tensor_224"].detach().cpu(),
                }
                if not bypass_cache:
                    self._vision_cache.set(vision_key, vision_bundle)
            vision_ms = (
                round((time.perf_counter() - t_vis) * 1000, 2)
                if vision_cache_hit
                else float(vision_bundle.get("pure_gpu_ms", round((time.perf_counter() - t_vis) * 1000, 2)))
            )

            # Fast pre-Decision preliminary fusion for immediate Stage-1 streaming (~350ms!)
            prelim = self.ensemble.fuse_predictions(
                vision_bundle=vision_bundle,
                gemini_decision=None,
                gemini_weight=AI_RUNTIME_CONFIG.decision_fusion_weight,
                dicom_view=dicom_view,
            )
            prelim_preds, prelim_pos = self._assemble_predictions(prelim["scores"], prelim["meta"], thr)
            seg_masks = prelim["seg_masks"]
            display_rgb_512 = prelim["display_rgb_512"]
            tensor_in = prelim["tensor_224"]
            active_backbones = prelim["active_backbones"]

            # Generate Stage-1 Grad-CAM++ in ONE forward pass while Gemini Decision 1 is still in-flight!
            t_cam = time.perf_counter()
            hm_map: dict[str, dict[str, Any]] = {}
            prelim_heatmaps: list[dict[str, Any]] = []
            if generate_heatmaps:
                cams_targets = prelim_pos[:4] if prelim_pos else prelim_preds[:1]
                prelim_heatmaps = self._generate_heatmaps_fast(
                    tensor_in, display_rgb_512, cams_targets, seg_masks, colormap
                )
                hm_map = {h["pathology"]: h for h in prelim_heatmaps}
            gradcam_ms = round((time.perf_counter() - t_cam) * 1000, 2)

            # Emit Stage 1 (`vision`) immediately so Lightbox + CTR + Grad-CAM++ appear in <0.5s!
            yield {
                "event": "vision",
                "data": {
                    "predictions": prelim_preds,
                    "positives": prelim_pos,
                    "heatmaps": prelim_heatmaps,
                    "anatomy": prelim["anatomy"],
                    "ensemble_info": {
                        "mode": self.ensemble_mode,
                        "active_backbones": active_backbones,
                        "tta_enabled": tta,
                        "segmentation_enabled": bool(prelim["anatomy"]),
                        "gemini_decision_active": False,
                        "device": str(self.device),
                    },
                    "timings": {
                        "vision_ms": vision_ms,
                        "gradcam_ms": gradcam_ms,
                        "core_total_ms": round(vision_ms + gradcam_ms, 2),
                    },
                    "vision_latency_ms": round(vision_ms + gradcam_ms, 2),
                },
            }

            # 4. Collect concurrent Gemini Decision 1 & run final projection-aware consensus fusion
            if dec_future is not None:
                try:
                    decision_summary = dec_future.result(timeout=12.0)
                except Exception as exc:
                    decision_summary = {"active": False, "status": "timeout", "latency_ms": 0.0, "message": str(exc)}

            final_ens = self.ensemble.fuse_predictions(
                vision_bundle=vision_bundle,
                gemini_decision=decision_summary,
                gemini_weight=AI_RUNTIME_CONFIG.decision_fusion_weight,
                dicom_view=dicom_view,
            )
            scores = final_ens["scores"]
            per_pathology_meta = final_ens["meta"]
            anatomy = final_ens["anatomy"]
            all_preds, positives = self._assemble_predictions(scores, per_pathology_meta, thr)

            # Reuse Stage-1 heatmaps and only compute any newly promoted positive class!
            t_cam2 = time.perf_counter()
            if generate_heatmaps:
                final_targets = positives[:4] if positives else all_preds[:1]
                heatmaps = self._generate_heatmaps_fast(
                    tensor_in,
                    display_rgb_512,
                    final_targets,
                    seg_masks,
                    colormap,
                    existing_map=hm_map,
                )
            else:
                heatmaps = []
            gradcam_ms = round(gradcam_ms + (time.perf_counter() - t_cam2) * 1000, 2)
            core_vision_total_ms = round(vision_ms + gradcam_ms, 2)

            if decision_summary.get("active"):
                yield {
                    "event": "consensus",
                    "data": {
                        "predictions": all_preds,
                        "positives": positives,
                        "heatmaps": heatmaps,
                        "anatomy": anatomy,
                        "decision_summary": decision_summary,
                        "ensemble_info": {
                            "mode": self.ensemble_mode,
                            "active_backbones": active_backbones,
                            "tta_enabled": tta,
                            "segmentation_enabled": bool(anatomy),
                            "gemini_decision_active": True,
                            "device": str(self.device),
                        },
                        "timings": {
                            "vision_ms": vision_ms,
                            "gradcam_ms": gradcam_ms,
                            "core_total_ms": core_vision_total_ms,
                            "decision_api_ms": float(decision_summary.get("latency_ms", 0.0)),
                        },
                        "vision_latency_ms": core_vision_total_ms,
                    },
                }
        else:
            # Local custom checkpoint fallback
            if dec_future is not None:
                try:
                    decision_summary = dec_future.result(timeout=12.0)
                except Exception:
                    pass
            tensor, display_rgb_512, _ = self.preprocess_image(pil_img)
            tensor_in = tensor.to(self.device)
            with torch.inference_mode():
                raw_out = self.model(tensor_in)
                probs = torch.sigmoid(raw_out).squeeze(0).cpu().numpy()
            scores = {
                p: round(max(0.001, min(0.999, float(probs[i]) if i < len(probs) else 0.05)), 4)
                for i, p in enumerate(PATHOLOGIES)
            }
            vision_ms = round((time.perf_counter() - t_vis) * 1000, 2)
            anatomy = {}
            seg_masks = {}
            active_backbones = ["densenet121_local"]
            all_preds, positives = self._assemble_predictions(scores, {}, thr)

            t_cam = time.perf_counter()
            if generate_heatmaps:
                cams_targets = positives[:4] if positives else all_preds[:1]
                heatmaps = self._generate_heatmaps_fast(
                    tensor_in, display_rgb_512, cams_targets, seg_masks, colormap
                )
            else:
                heatmaps = []
            gradcam_ms = round((time.perf_counter() - t_cam) * 1000, 2)
            core_vision_total_ms = round(vision_ms + gradcam_ms, 2)

        # 5. Multi-Stage Clinical Report Synthesis (Gemini Flash -> Groq -> ACR Clinical NLP)
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

        payload: dict[str, Any] = {
            "predictions": all_preds,
            "positives": positives,
            "heatmaps": heatmaps,
            "anatomy": anatomy,
            "decision_summary": decision_summary,
            "report": report,
            "cache_hit": False,
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
            "inference_time_ms": elapsed_ms,
            "vision_latency_ms": core_vision_total_ms,
        }

        if not bypass_cache:
            self._result_cache.set(full_key, copy.deepcopy(payload))

        yield {"event": "complete", "data": payload}

    # ------------------------------------------------------------------
    # Synchronous Inference API (100% Backward-Compatible)
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
        generate_heatmaps: bool = True,
        dicom_view: str | None = None,
        bypass_cache: bool = False,
    ) -> dict[str, Any]:
        """Run full 4-backbone transfer diagnosis, non-chiral TTA, PSPNet CTR, Grad-CAM++, and report synthesis."""
        final_payload: dict[str, Any] | None = None
        for evt in self.predict_stream(
            image_input=image_input,
            threshold=threshold,
            tta=tta,
            patient=patient,
            use_gemini_decision=use_gemini_decision,
            use_llm=use_llm,
            colormap=colormap,
            run_segmentation=run_segmentation,
            generate_heatmaps=generate_heatmaps,
            dicom_view=dicom_view,
            bypass_cache=bypass_cache,
        ):
            if evt.get("event") == "complete":
                final_payload = evt["data"]
        if final_payload is None:
            raise RuntimeError("DiagnosisPipeline produced no final prediction payload.")
        return final_payload

    def generate_single_heatmap(
        self,
        image_input: bytes | Image.Image,
        pathology_name: str,
        colormap: str = "turbo",
    ) -> dict[str, Any] | None:
        """Generate a Grad-CAM++ heatmap + pathology-aware anatomical zone for one pathology on demand."""
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
            hs_meta = extract_hotspot_metadata(cam, pathology=pathology_name)
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
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            out = torch.sigmoid(self.model(batch).float())
        return out.cpu()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if hasattr(self, "_gradcam") and self._gradcam:
            self._gradcam.close()
        if hasattr(self, "_io_pool") and self._io_pool:
            self._io_pool.shutdown(wait=False)

    def __enter__(self) -> "DiagnosisPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
