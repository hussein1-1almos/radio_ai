"""MedMO-4B-Next inference adapter for the existing FastAPI app.

This module deliberately contains no legacy MedGemma engine/class. It exposes the
small functional interface app.py expects. Output is a draft for clinician review,
not a validated diagnostic result. No lesion localization is claimed.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/srv/models/MedMO-4B-Next"))
NUM_THREADS = max(1, int(os.getenv("NUM_THREADS", str(os.cpu_count() or 4))))
MAX_NEW_TOKENS = max(64, int(os.getenv("MAX_NEW_TOKENS", "512")))
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() in {"1", "true", "yes"}

# app.py uses this for /health/ready. It records the loaded model, not a validated detector.
_cache: dict[str, Any] = {}
_model = None
_processor = None
_load_lock = threading.Lock()

SYSTEM_PROMPT = """You are assisting with drafting a radiology report from ONE medical image.
Return only a valid JSON object with these keys:
{
  "examination": string,
  "technique": string,
  "summary": string,
  "limitations": [string, ...],
  "impression": [string, ...],
  "uncertainty": "High" | "Moderate" | "Low",
  "needs_additional_images": boolean
}
Rules:
- Describe only findings that are actually visible in the supplied image; do not invent history, measurements, or findings.
- A single image/instance is not a complete study. Explicitly state this limitation.
- If image quality, modality, orientation, or anatomy prevents reliable interpretation, say so and use high uncertainty.
- Do not claim that an abnormality is absent unless it is clearly assessable in this image.
- Do not provide a definitive diagnosis, autonomous triage, treatment, or urgency determination.
- Do not provide bounding boxes or claim lesion localization.
- The output is an unvalidated preliminary draft requiring review by a qualified radiologist.
- Use concise professional language. JSON strings must be escaped correctly.
"""


def _load_model():
    """Load local model once. CPU float32 avoids requiring a CUDA device."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    with _load_lock:
        if _model is not None and _processor is not None:
            return _model, _processor
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"MODEL_PATH does not exist: {MODEL_PATH}")
        torch.set_num_threads(NUM_THREADS)
        _processor = AutoProcessor.from_pretrained(
            str(MODEL_PATH), local_files_only=True, trust_remote_code=TRUST_REMOTE_CODE
        )
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.float32,
            device_map="cpu",
            local_files_only=True,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
        _model.eval()
        _cache["medmo-4b-next"] = {
            "name": "MedMO-4B-Next",
            "status": "loaded",
            "device": "cpu",
        }
        return _model, _processor


def status() -> dict[str, Any]:
    return {
        "model": "MedMO-4B-Next",
        "model_path_exists": MODEL_PATH.exists(),
        "loaded": _model is not None,
        "device": "cpu",
        "validated": False,
    }


def coverage() -> dict[str, Any]:
    """Report configured model availability without implying clinical validation."""
    return {
        "model": "MedMO-4B-Next",
        "device": "cpu",
        "validated": False,
        "clinical_use": "research/testing only; radiologist review required",
        "supported_input": "single 2D DICOM instance converted to an image",
        "localization": False,
        "sites": {},
        "limitations": [
            "No local validation or regulatory clearance is asserted.",
            "No bounding-box/lesion-localization model is installed by this adapter.",
            "Single-instance input is not a substitute for the complete imaging study.",
        ],
    }


def unload_idle(idle_seconds: float = 300) -> None:
    """Intentionally retain the model in RAM to avoid unsafe concurrent unloads."""
    return None


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:].lstrip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model did not return a JSON report")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model report JSON must be an object")
    return value


def analyze(img: Image.Image, arr, meta: dict[str, Any], site: str = "") -> dict[str, Any]:
    """Generate a preliminary report using the local MedMO vision-language model."""
    model, processor = _load_model()
    image = img.convert("RGB")
    # Qwen3-VL's chat template accepts a PIL image as the image content.
    user_text = (
        "Draft a cautious radiology report for this single image. "
        f"Provided non-identifying study context: modality={meta.get('modality', 'Unknown')}; "
        f"body_part={meta.get('body_part', 'Unknown')}; site_context={site or 'not supplied'}. "
        "Do not infer missing clinical history. Return the required JSON only."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = {key: value.to("cpu") if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    answer_tokens = generated[:, prompt_len:]
    decoded = processor.batch_decode(answer_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    parsed = _extract_json(decoded)

    def string_value(key: str, default: str = "") -> str:
        value = parsed.get(key, default)
        return value.strip() if isinstance(value, str) else default

    def string_list(key: str, default: list[str] | None = None) -> list[str]:
        value = parsed.get(key, default or [])
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return default or []
        return [str(item).strip() for item in value if str(item).strip()]

    uncertainty = string_value("uncertainty", "High").capitalize()
    if uncertainty not in {"High", "Moderate", "Low"}:
        uncertainty = "High"
    limitations = string_list("limitations")
    limitations.append("AI-generated, unvalidated draft from a single 2D image; radiologist review is required.")
    return {
        "examination": string_value("examination", f"{meta.get('modality', 'Unknown')} imaging"),
        "technique": string_value("technique", "Single 2D DICOM instance; full acquisition protocol unavailable."),
        "findings": {
            "summary": string_value("summary", "The model did not provide a reliable summary."),
            "limitations": list(dict.fromkeys(limitations)),
            "needs_additional_images": bool(parsed.get("needs_additional_images", True)),
            "possible_urgent_finding": None,
            "bounding_box": [],
            "flagged": [],
        },
        "impression": string_list("impression", ["Indeterminate; requires review by a qualified radiologist."]),
        "uncertainty": uncertainty,
        "annotations": [],
        "region_hints": [],
        "annotation_status": "localization_unavailable",
        "annotated_image_available": False,
        "annotated_image_url": None,
        "models": [{
            "name": "MedMO-4B-Next",
            "kind": "vision_language_model",
            "version": "local checkpoint",
            "validated": False,
            "status": "ran",
            "detects": "Image-conditioned draft text only; localization is not implemented",
            "detail": "CPU float32 inference. Output is not clinically validated.",
        }],
        "recommendations": ["Have a qualified radiologist review the original study and this draft."],
        "sources": [],
        "review_required": True,
    }


def build_report(result: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Keep the response contract expected by app.py."""
    report = dict(result)
    report["metadata"] = dict(meta)
    return report


def narrate(report: dict[str, Any]):
    """No second text model is used; report text already comes from MedMO."""
    return None


def annotate(img: Image.Image, annotations: list[dict[str, Any]]) -> Image.Image:
    """No boxes are generated, so return an unchanged copy if called."""
    return img.copy()
