"""Self-hosted MedMO DICOM draft-report API prototype.
Not clinically validated. Verify the inference adapter against the exact official
MedMO repository revision before use. Always require radiologist review.
"""
import io, json, logging, os, secrets, uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pydicom
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Header, Depends, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, ValidationError

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("medmo-api")

API_KEY = os.getenv("API_KEY", "").strip()
MODEL_PATH = os.getenv("MODEL_PATH", "/srv/models/MedMO-8B-Next")
DEVICE = os.getenv("DEVICE", "cpu").lower()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "33554432"))
MAX_IMAGE_EDGE = int(os.getenv("MAX_IMAGE_EDGE", "1536"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
torch.set_num_threads(int(os.getenv("NUM_THREADS", str(max(1, os.cpu_count() or 1)))))
ANNOTATION_DIR = Path(os.getenv("ANNOTATION_DIR", "./data/annotations"))
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MedMO DICOM Draft Report API", version="0.1.0")
state: dict[str, Any] = {"loaded": False, "model": None, "processor": None, "error": None}


class Findings(BaseModel):
    summary: str
    limitations: list[str] = Field(default_factory=list)
    needs_additional_images: bool = False
    possible_urgent_finding: bool = False


class Box(BaseModel):
    label: str
    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=1)
    y_max: int = Field(ge=1)
    status: str = "unverified"


class Annotations(BaseModel):
    available: bool = False
    image_url: Optional[str] = None
    coordinate_system: str = "original_image_pixels"
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    boxes: list[Box] = Field(default_factory=list)
    status: str = "not_available"


class DraftReport(BaseModel):
    findings: Findings
    impression: str
    uncertainty: str
    review_required: bool = True
    annotations: Annotations = Field(default_factory=Annotations)


def auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_KEY or API_KEY in {"replace-with-a-long-random-secret", "change-me"}:
        raise HTTPException(503, "API_KEY is not configured securely")
    if authorization is None or not secrets.compare_digest(authorization, f"Bearer {API_KEY}"):
        raise HTTPException(401, "Invalid or missing bearer token")


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready", dependencies=[Depends(auth)])
def ready():
    return {"status": "ready" if state["loaded"] else "model_not_loaded",
            "model_loaded": state["loaded"], "device": DEVICE, "model_path": MODEL_PATH,
            "last_load_error": state["error"]}


def load_model():
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"MODEL_PATH does not exist: {MODEL_PATH}")
        trust = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=trust)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=trust,
            torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
            low_cpu_mem_usage=True)
        model.to(DEVICE).eval()
        state.update(loaded=True, model=model, processor=processor, error=None)
        log.info("Model loaded from %s on %s", MODEL_PATH, DEVICE)
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("Model loading failed")
        raise


@app.on_event("startup")
def startup():
    if os.getenv("LOAD_MODEL_ON_STARTUP", "true").lower() == "true":
        load_model()


def dicom_to_image(raw: bytes):
    try:
        ds = pydicom.dcmread(io.BytesIO(raw), force=False)
    except Exception as exc:
        raise HTTPException(400, f"Invalid DICOM file: {exc}") from exc
    if not hasattr(ds, "PixelData"):
        raise HTTPException(422, "DICOM contains no pixel data")
    if int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise HTTPException(422, "Prototype supports single-frame DICOM only")
    try:
        pixels = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise HTTPException(422, f"Cannot decode DICOM pixels; check transfer-syntax codecs: {exc}") from exc
    if pixels.ndim != 2:
        raise HTTPException(422, "Only single-channel 2D DICOM images are supported")
    pixels = pixels * float(getattr(ds, "RescaleSlope", 1.0))
    pixels += float(getattr(ds, "RescaleIntercept", 0.0))
    try:
        center = float(ds.WindowCenter[0] if hasattr(ds.WindowCenter, "__getitem__") else ds.WindowCenter)
        width = float(ds.WindowWidth[0] if hasattr(ds.WindowWidth, "__getitem__") else ds.WindowWidth)
        if width <= 0:
            raise ValueError()
        lo, hi = center - width / 2, center + width / 2
    except Exception:
        lo, hi = np.percentile(pixels, [1, 99])
        if hi <= lo:
            lo, hi = float(pixels.min()), float(pixels.max()) + 1
    pixels = np.clip((pixels - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = 1 - pixels
    image = Image.fromarray((pixels * 255).astype(np.uint8)).convert("RGB")
    metadata = {"modality": str(getattr(ds, "Modality", "UNKNOWN")),
                "rows": image.height, "columns": image.width}
    scale = min(1.0, MAX_IMAGE_EDGE / max(image.size))
    if scale < 1:
        image = image.resize((max(1, round(image.width * scale)),
                              max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    return image, metadata


PROMPT = """Draft a preliminary radiology report from ONLY the supplied 2D image.
Do not invent findings or claim to have assessed unseen slices or a whole study.
State limitations and uncertainty. If localization is not reliable, return no boxes.
Every output requires qualified radiologist review; this is not a final diagnosis.
Return JSON only:
{
 "findings":{"summary":"string","limitations":["string"],
 "needs_additional_images":false,"possible_urgent_finding":false},
 "impression":"string","uncertainty":"string","review_required":true,
 "boxes":[{"label":"candidate finding","x_min":0,"y_min":0,"x_max":1,"y_max":1}]
}
Box coordinates are integer pixels on the supplied image, origin top-left.
Only return specific, genuinely localized candidate regions. Never use a generic
whole-image box. Boxes are unverified candidates, not confirmed abnormalities.
"""


def run_medmo(image: Image.Image, metadata: dict) -> dict:
    """Generic adapter; verify/adjust to the installed MedMO-Next inference example."""
    if not state["loaded"]:
        raise HTTPException(503, "Model is not loaded")
    processor, model = state["processor"], state["model"]
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT + "\nMetadata: " + json.dumps(metadata)}
    ]}]
    try:
        if hasattr(processor, "apply_chat_template"):
            prompt = processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        else:
            inputs = processor(text=PROMPT, images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        if "input_ids" in inputs:
            ids = ids[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model did not return a JSON object")
        return json.loads(text[start:end + 1])
    except Exception as exc:
        log.exception("Inference/output parsing failed")
        raise HTTPException(502, f"MedMO inference/output parsing failed: {type(exc).__name__}: {exc}") from exc


def validate_boxes(raw_boxes, inference_size, original_size):
    valid, original_boxes = [], []
    sx, sy = original_size[0] / inference_size[0], original_size[1] / inference_size[1]
    if not isinstance(raw_boxes, list):
        return valid
    for item in raw_boxes:
        try:
            x1 = max(0, int(round(float(item["x_min"]) * sx)))
            y1 = max(0, int(round(float(item["y_min"]) * sy)))
            x2 = min(original_size[0], int(round(float(item["x_max"]) * sx)))
            y2 = min(original_size[1], int(round(float(item["y_max"]) * sy)))
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) * (y2 - y1) > 0.85 * original_size[0] * original_size[1]:
                continue
            valid.append(Box(label=str(item.get("label", "candidate finding"))[:120],
                             x_min=x1, y_min=y1, x_max=x2, y_max=y2,
                             status="unverified"))
        except (KeyError, TypeError, ValueError, ValidationError):
            continue
    return valid


def save_annotation(image: Image.Image, boxes: list[Box]) -> Optional[str]:
    if not boxes:
        return None
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width = max(2, min(out.size) // 250)
    for b in boxes:
        draw.rectangle((b.x_min, b.y_min, b.x_max, b.y_max),
                       outline=(255, 210, 0), width=width)
        draw.text((b.x_min, max(0, b.y_min - 18)),
                  f"UNVERIFIED: {b.label}", fill=(255, 210, 0))
    image_id = uuid.uuid4().hex
    out.save(ANNOTATION_DIR / f"{image_id}.png", format="PNG")
    return image_id


@app.post("/v1/dicom/draft-report", response_model=DraftReport)
async def draft_report(file: UploadFile = File(...), _: None = Depends(auth)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds MAX_UPLOAD_BYTES")
    if not raw:
        raise HTTPException(400, "Empty upload")
    image, metadata = dicom_to_image(raw)
    original_size = image.size
    infer_image = image
    scale = min(1.0, MAX_IMAGE_EDGE / max(image.size))
    if scale < 1:
        infer_image = image.resize((max(1, round(image.width * scale)),
                                    max(1, round(image.height * scale))),
                                   Image.Resampling.LANCZOS)
    result = run_medmo(infer_image, metadata)
    try:
        findings = Findings.model_validate(result["findings"])
        impression = str(result["impression"])
        uncertainty = str(result["uncertainty"])
    except (KeyError, TypeError, ValidationError) as exc:
        raise HTTPException(502, f"Model output failed schema validation: {exc}") from exc

    boxes = validate_boxes(result.get("boxes", []), infer_image.size, original_size)
    image_id = save_annotation(image, boxes)
    return DraftReport(
        findings=findings, impression=impression, uncertainty=uncertainty,
        review_required=True,
        annotations=Annotations(
            available=bool(image_id),
            image_url=f"/v1/dicom/annotations/{image_id}" if image_id else None,
            coordinate_system="original_image_pixels",
            image_width=original_size[0], image_height=original_size[1],
            boxes=boxes,
            status="unverified_candidate_localization" if boxes else "no_validated_localization",
        ),
    )


@app.get("/v1/dicom/annotations/{image_id}", dependencies=[Depends(auth)])
def get_annotation(image_id: str):
    if not image_id.isalnum() or len(image_id) != 32:
        raise HTTPException(404, "Annotation not found")
    path = ANNOTATION_DIR / f"{image_id}.png"
    if not path.is_file():
        raise HTTPException(404, "Annotation not found")
    return FileResponse(path, media_type="image/png", filename=f"{image_id}.png")
