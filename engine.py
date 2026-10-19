from __future__ import annotations
import ast, json, logging, os, re, threading, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

log = logging.getLogger("medmo.engine")
MODEL_PATH = os.getenv("MODEL_PATH", "/srv/models/MedMO-4B-Next")
DEVICE = os.getenv("DEVICE", "cpu").lower()
NUM_THREADS = int(os.getenv("NUM_THREADS", "8"))
MAX_NEW_TOKENS = int(os.getenv("LOCALIZATION_MAX_NEW_TOKENS", "256"))
BOX_SCALE = float(os.getenv("BOX_COORDINATE_SCALE", "1000"))
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() in {"1", "true", "yes"}
MODEL_VERSION = os.getenv("MODEL_VERSION", "MedMO-4B-Next")
_lock = threading.Lock()
_load_lock = threading.Lock()
_model = _processor = None
_last_used = 0.0
_last_error = None
if NUM_THREADS > 0:
    torch.set_num_threads(NUM_THREADS)

def _load_model():
    global _model, _processor, _last_used, _last_error
    if DEVICE != "cpu":
        raise RuntimeError("This deployment supports CPU only; set DEVICE=cpu.")
    if _model is not None and _processor is not None:
        _last_used = time.time()
        return _model, _processor
    with _load_lock:
        if _model is not None and _processor is not None:
            _last_used = time.time()
            return _model, _processor
        path = Path(MODEL_PATH)
        if not path.is_dir():
            raise FileNotFoundError(f"MODEL_PATH does not exist: {path}")
        try:
            processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=TRUST_REMOTE_CODE, local_files_only=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(path), torch_dtype=torch.float32, device_map={"": "cpu"},
                trust_remote_code=TRUST_REMOTE_CODE, local_files_only=True
            )
            model.eval()
            _processor, _model = processor, model
            _last_used = time.time()
            _last_error = None
            log.info("Loaded %s from %s on CPU", MODEL_VERSION, path)
            return _model, _processor
        except Exception as exc:
            _last_error = repr(exc)
            log.exception("Model load failed")
            raise

def status():
    return {"loaded": _model is not None, "model_path": MODEL_PATH, "device": DEVICE,
            "model_version": MODEL_VERSION, "last_error": _last_error}

def coverage():
    return {"device": DEVICE, "detectors": [{
        "name": MODEL_VERSION, "kind": "vision_language_model", "version": MODEL_VERSION,
        "validated": False, "status": "ran" if _model is not None else "not_applicable",
        "detects": "Model-proposed candidate regions", "detail": "General-purpose model; site/modality-specific performance is unvalidated."
    }], "localization": {"enabled": True, "coordinate_scale_assumption": BOX_SCALE,
                         "coordinate_scale_verified": False}}

def unload_idle(idle_seconds: float):
    global _model, _processor
    if idle_seconds <= 0 or _model is None or time.time() - _last_used < idle_seconds:
        return
    if _load_lock.acquire(blocking=False):
        try:
            if _model is not None and time.time() - _last_used >= idle_seconds:
                _model = _processor = None
                log.info("Unloaded idle model")
        finally:
            _load_lock.release()

def _run_model(image: Image.Image, prompt: str) -> str:
    global _last_used
    model, processor = _load_model()
    image = image.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: (v.to("cpu") if hasattr(v, "to") else v) for k, v in inputs.items()}
    with _lock, torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
    ids = inputs.get("input_ids")
    if ids is not None:
        output = output[:, ids.shape[1]:]
    text = processor.batch_decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    _last_used = time.time()
    return text.strip()

def _boxes(value: Any):
    result = []
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(x, (int, float)) for x in value):
            result.append([float(x) for x in value])
        else:
            for x in value:
                result.extend(_boxes(x))
    return result

def _parse_payload(payload: str):
    for fn in (json.loads, ast.literal_eval):
        try:
            found = _boxes(fn(payload.strip()))
            if found:
                return found
        except Exception:
            pass
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", payload)]
    return [nums[i:i+4] for i in range(0, len(nums)-3, 4)]

def parse_localization_output(raw: str):
    matches = re.findall(r"<box\b[^>]*>(.*?)</box>", raw, flags=re.I|re.S)
    raw_boxes = []
    for match in matches:
        raw_boxes.extend(_parse_payload(match))
    if not raw_boxes and re.search(r"<box\b", raw, re.I):
        tail = re.split(r"<box\b[^>]*>", raw, maxsplit=1, flags=re.I)[-1]
        tail = re.split(r"</box>", tail, maxsplit=1, flags=re.I)[0]
        raw_boxes.extend(_parse_payload(tail))
    parsed = []
    for b in raw_boxes:
        if len(b) != 4 or not all(np.isfinite(v) for v in b) or BOX_SCALE <= 0:
            continue
        if min(b) < 0 or max(b) > BOX_SCALE:
            continue
        x1, y1, x2, y2 = b
        x1, x2 = sorted((x1/BOX_SCALE, x2/BOX_SCALE))
        y1, y2 = sorted((y1/BOX_SCALE, y2/BOX_SCALE))
        if x2-x1 < .002 or y2-y1 < .002:
            continue
        parsed.append([round(max(0,min(1,x1)),6), round(max(0,min(1,y1)),6),
                       round(max(0,min(1,x2)),6), round(max(0,min(1,y2)),6)])
    return parsed, ("localized" if parsed else
                    "localization_unavailable" if matches or re.search(r"<box\b", raw, re.I)
                    else "none_detected")

def localize_image(img: Image.Image):
    prompt = (
        "Inspect this medical image for visible abnormalities and candidate regions. "
        "If you can localize any candidate, emit model <box> tags with coordinates "
        "in [x_min,y_min,x_max,y_max] order, range 0-1000, relative to the displayed image. "
        "Use only regions supported by the image. If no region can be localized, say so. "
        "Do not give treatment advice or claim a diagnosis. This is an unvalidated localization proposal."
    )
    raw = _run_model(img, prompt)
    boxes, state = parse_localization_output(raw)
    anns = [{
        "id": f"medmo-{i}", "label": "AI-proposed candidate region; review required",
        "box": b, "box_format": "xyxy_normalized_0_1", "score": None,
        "score_calibrated": False, "source_model": MODEL_VERSION,
        "model_version": MODEL_VERSION, "validated": False,
        "image_ref": {"coordinate_space": "resized_analysis_image"}
    } for i, b in enumerate(boxes, 1)]
    return {"annotations": anns, "status": state, "raw_output": raw}

def analyze(img: Image.Image, arr, meta: dict, site: str = ""):
    modality = str(meta.get("modality", "Unknown"))
    try:
        loc = localize_image(img)
        anns = loc["annotations"]
        legacy = [[round(a["box"][1]*1000,2), round(a["box"][0]*1000,2),
                   round(a["box"][3]*1000,2), round(a["box"][2]*1000,2)] for a in anns]
        summary = (f"Model proposed {len(anns)} candidate region(s); these are not confirmed findings."
                   if anns else "No usable localization box was returned. This does not establish a normal image.")
        return {
            "examination": f"{modality} single-image localization",
            "technique": "Single 2D DICOM instance converted to an 8-bit display image",
            "findings": {"summary": summary, "limitations": [
                "AI output is unvalidated and is not a diagnostic report.",
                "No box does not exclude disease or establish a normal study.",
                "The assumed 0-1000 coordinate convention has not been verified for this checkpoint.",
                "Only one 2D image was analysed; no 3D volume/series interpretation was performed."
            ], "needs_additional_images": True, "possible_urgent_finding": None,
                "bounding_box": legacy, "flagged": []},
            "impression": [], "uncertainty": "High", "annotations": anns,
            "region_hints": [], "annotation_status": loc["status"],
            "annotated_image_available": False, "annotated_image_url": None,
            "models": [{"name": MODEL_VERSION, "kind": "vision_language_model",
                "version": MODEL_VERSION, "validated": False, "status": "ran",
                "detects": "Candidate regions proposed by a general-purpose model",
                "detail": "Unvalidated; coordinate convention requires verification."}],
            "recommendations": ["Radiologist review is required before any clinical use."],
            "sources": [], "review_required": True, "_raw_output": loc["raw_output"]
        }
    except Exception as exc:
        log.exception("Localization failed")
        return {
            "examination": f"{modality} single-image localization", "technique": "Single 2D DICOM instance",
            "findings": {"summary": "Localization failed; no conclusion about the image can be drawn.",
                "limitations": ["Inference/output parsing failed.", "Radiologist review is required."],
                "needs_additional_images": True, "possible_urgent_finding": None, "bounding_box": [], "flagged": []},
            "impression": [], "uncertainty": "High", "annotations": [], "region_hints": [],
            "annotation_status": "analysis_failed", "annotated_image_available": False,
            "annotated_image_url": None, "models": [{"name": MODEL_VERSION,
                "kind": "vision_language_model", "version": MODEL_VERSION, "validated": False,
                "status": "failed", "detects": "None", "detail": repr(exc)[:400]}],
            "recommendations": ["Radiologist review is required; do not interpret failure as normal."],
            "sources": [], "review_required": True
        }

def build_report(result: dict, meta: dict):
    result = dict(result)
    result.pop("_raw_output", None)
    result["metadata"] = meta
    result["review_required"] = True
    return result

def narrate(report: dict):
    return None

def annotate(img: Image.Image, annotations: list[dict]):
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for ann in annotations:
        box = ann.get("box", [])
        if len(box) != 4:
            continue
        try:
            x1,y1,x2,y2 = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        x1,x2 = sorted((max(0,min(w-1,round(x1*w))), max(0,min(w-1,round(x2*w)))))
        y1,y2 = sorted((max(0,min(h-1,round(y1*h))), max(0,min(h-1,round(y2*h)))))
        draw.rectangle((x1,y1,x2,y2), outline=(255,0,0), width=max(2,round(min(w,h)/300)))
        draw.text((x1,max(0,y1-14)), str(ann.get("label","AI candidate"))[:70], fill=(255,0,0))
    return out
