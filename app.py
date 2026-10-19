from __future__ import annotations
import base64, io, json, os, secrets, threading
from contextlib import asynccontextmanager
from typing import Any
from pathlib import Path

import numpy as np
import pydicom
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from pydicom.pixels import apply_modality_lut, apply_voi_lut

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def resolve_project_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path.resolve())


MODEL_PATH = resolve_project_path(
    os.getenv("MODEL_PATH", "models/medgemma-4b-it-Q4_K_M.gguf")
)
MMPROJ_PATH = resolve_project_path(
    os.getenv("MMPROJ_PATH", "models/mmproj-F16.gguf")
)
CHAT_HANDLER_CLASS = os.getenv("CHAT_HANDLER_CLASS", "MTMDChatHandler")
API_KEY = os.getenv("API_KEY", "")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))
MAX_IMAGE_EDGE = int(os.getenv("MAX_IMAGE_EDGE", "1536"))
N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", str(max(1, (os.cpu_count() or 2) - 1))))
N_BATCH = int(os.getenv("N_BATCH", "256"))
ALLOWED_MODALITIES = {"CR", "DX", "DR", "CT", "MR", "MRI", "US"}
llm = None
model_lock = threading.Lock()


class Findings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str = Field(default="No summary generated.")
    limitations: list[str] = Field(default_factory=list)
    needs_additional_images: bool = Field(default=False)
    possible_urgent_finding: bool = Field(default=False)
    bounding_box: list[float] = Field(
        default_factory=list,
        description="Normalized coordinates [ymin, xmin, ymax, xmax] from 0 to 1000 if a focal lesion is found."
    )


class DraftReport(BaseModel):
    model_config = ConfigDict(extra="ignore")
    metadata: dict[str, Any] = Field(default_factory=dict)
    findings: Findings = Field(default_factory=Findings)
    impression: list[str] = Field(default_factory=list)
    uncertainty: str = Field(default="Unknown")
    review_required: bool = True


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not API_KEY:
        raise HTTPException(503, "API authentication is not configured")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not supplied or not secrets.compare_digest(supplied, API_KEY):
        raise HTTPException(401, "Invalid authentication credentials")


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("Pixel data contains no finite values")
    lo, hi = np.percentile(arr[finite], [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(arr[finite])), float(np.max(arr[finite]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (np.clip(arr, lo, hi) - lo) / (hi - lo) * 255
    out[~finite] = 0
    return out.astype(np.uint8)


def dicom_to_png(raw: bytes) -> tuple[bytes, dict[str, Any]]:
    try:
        ds = pydicom.dcmread(io.BytesIO(raw), force=False)
    except Exception as exc:
        raise ValueError("Upload is not a readable DICOM file") from exc
    modality = str(getattr(ds, "Modality", "")).upper().strip()
    if modality not in ALLOWED_MODALITIES:
        raise ValueError(f"Unsupported modality: {modality or 'missing'}")
    if "PixelData" not in ds:
        raise ValueError("DICOM contains no PixelData")
    if int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise ValueError("Only single-frame 2D DICOM instances are supported")
    try:
        pixels = ds.pixel_array
    except Exception as exc:
        raise ValueError("Cannot decode pixels; an additional DICOM codec may be required") from exc
    if pixels.ndim not in (2, 3):
        raise ValueError("Only 2D images are supported")
    if pixels.ndim == 3:
        if pixels.shape[-1] not in (3, 4):
            raise ValueError("Unsupported color pixel layout")
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "PALETTE COLOR":
            raise ValueError("Palette-color DICOM is not supported")
        if pixels.dtype != np.uint8:
            pixels = to_uint8(pixels)
        image = Image.fromarray(pixels[..., :3]).convert("RGB")
    else:
        try:
            pixels = apply_modality_lut(pixels, ds)
        except Exception:
            pixels = pixels.astype(np.float32)
        try:
            gray = to_uint8(np.asarray(apply_voi_lut(pixels, ds), dtype=np.float32))
        except Exception:
            if modality == "CT":
                try:
                    wc = float(ds.WindowCenter[0] if hasattr(ds.WindowCenter, "__len__") else ds.WindowCenter)
                    ww = float(ds.WindowWidth[0] if hasattr(ds.WindowWidth, "__len__") else ds.WindowWidth)
                except Exception:
                    wc, ww = float("nan"), float("nan")
                if np.isfinite(wc) and np.isfinite(ww) and ww > 1:
                    lo, hi = wc - ww / 2, wc + ww / 2
                    gray = (np.clip(pixels, lo, hi) - lo) / (hi - lo) * 255
                    gray = np.nan_to_num(gray).astype(np.uint8)
                else:
                    gray = to_uint8(pixels)
            else:
                gray = to_uint8(pixels)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            gray = 255 - gray

        image = Image.fromarray(gray, mode="L").convert("RGB")
    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)

    def safe_get(tag, default=None):
        val = getattr(ds, tag, default)
        if val is not None:
            if hasattr(val, "__iter__") and not isinstance(val, (str, bytes, dict)):
                return [float(x) if isinstance(x, (int, float, np.number)) else str(x) for x in val]
            if isinstance(val, (int, float, np.number)):
                return float(val) if isinstance(val, float) else int(val)
            return str(val).strip()
        return default

    metadata = {
        "patient_name": safe_get("PatientName", "Unknown"),
        "patient_id": safe_get("PatientID", "Unknown"),
        "patient_birth_date": safe_get("PatientBirthDate", "Unknown"),
        "patient_sex": safe_get("PatientSex", "Unknown"),
        "patient_age": safe_get("PatientAge", "Unknown"),
        "patient_weight": safe_get("PatientWeight", 0.0),
        "study_date": safe_get("StudyDate", "Unknown"),
        "study_time": safe_get("StudyTime", "Unknown"),
        "study_description": safe_get("StudyDescription", "Unknown"),
        "series_number": safe_get("SeriesNumber", 0),
        "series_description": safe_get("SeriesDescription", "Unknown"),
        "modality": modality,
        "accession_number": safe_get("AccessionNumber", "Unknown"),
        "manufacturer": safe_get("Manufacturer", "Unknown"),
        "manufacturer_model_name": safe_get("ManufacturerModelName", "Unknown"),
        "institution_name": safe_get("InstitutionName", "Unknown"),
        "rows": int(getattr(ds, "Rows", image.height)),
        "columns": int(getattr(ds, "Columns", image.width)),
        "slice_thickness": safe_get("SliceThickness", 0.0),
        "pixel_spacing": safe_get("PixelSpacing", [0.0, 0.0]),
        "magnetic_field_strength": safe_get("MagneticFieldStrength", 0.0),
        "kvp": safe_get("KVP", 0.0),
        "repetition_time": safe_get("RepetitionTime", 0.0),
        "echo_time": safe_get("EchoTime", 0.0),
        "scope": "single instance only"
    }
    return buf.getvalue(), metadata


REPORT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "findings": {"type": "object", "additionalProperties": False,
                     "properties": {
                         "summary": {"type": "string"},
                         "limitations": {"type": "array", "items": {"type": "string"}},
                         "needs_additional_images": {"type": "boolean"},
                         "possible_urgent_finding": {"type": "boolean"},
                         "bounding_box": {"type": "array", "items": {"type": "number"}}},
                     "required": ["summary", "limitations", "needs_additional_images", "possible_urgent_finding", "bounding_box"]},
        "impression": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": "string"}, "review_required": {"type": "boolean"}},
    "required": ["findings", "impression", "uncertainty", "review_required"]}


def load_model():
    global llm
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(MMPROJ_PATH):
        raise RuntimeError(f"Multimodal projector not found: {MMPROJ_PATH}")
    from llama_cpp import Llama
    import llama_cpp.llama_chat_format as formats
    handler_cls = getattr(formats, CHAT_HANDLER_CLASS, None)
    if handler_cls is None:
        raise RuntimeError(
            f"{CHAT_HANDLER_CLASS} is not available in this llama-cpp-python build. "
            "MedGemma is based on Gemma 3; do not silently substitute Gemma4ChatHandler."
        )
    if CHAT_HANDLER_CLASS == "MTMDChatHandler":
        handler = handler_cls(
            clip_model_path=MMPROJ_PATH,
            verbose=False,
            use_gpu=False,
        )
    else:
        handler = handler_cls(clip_model_path=MMPROJ_PATH, verbose=False)
    llm = Llama(model_path=MODEL_PATH, chat_handler=handler, n_ctx=N_CTX,
                n_threads=N_THREADS, n_batch=N_BATCH, n_gpu_layers=0, verbose=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    global llm
    llm = None


app = FastAPI(title="MedGemma DICOM Draft-Report API", version="0.2.4",
              description="Local llama-cpp-python integration with bounding box localization.",
              lifespan=lifespan)


def infer_sync(png: bytes, metadata: dict[str, Any]) -> DraftReport:
    if llm is None:
        raise RuntimeError("Model is not initialized")
    uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    prompt = (
        "Create a NON-FINAL draft for a qualified radiologist to review. This is one rendered "
        "image, not a complete study. Do not assert normality if evidence is insufficient. "
        "State limitations and uncertainty; do not give treatment recommendations. The possible "
        "urgent finding flag is only a model suggestion, must not be used as an autonomous alert "
        "or to rule out an emergency. If a specific focal abnormality or lesion is identified, "
        "provide its bounding box coordinates as [ymin, xmin, ymax, xmax] scaled from 0 to 1000. "
        "If no focal lesion is present, return an empty array []. "
        "Return JSON with findings.summary, findings.limitations, findings.needs_additional_images, "
        "findings.possible_urgent_finding, findings.bounding_box, impression array, uncertainty, "
        "and review_required. review_required must be true."
    )
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": uri}}]}]

    with model_lock:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=700,
            response_format={"type": "json_object", "schema": REPORT_SCHEMA},
        )

    choice = response["choices"][0]
    content = choice.get("message", {}).get("content") or ""
    finish_reason = choice.get("finish_reason")

    try:
        report = DraftReport.model_validate_json(content)
    except Exception:
        try:
            decoder = json.JSONDecoder()
            start = content.find("{")
            if start < 0:
                raise ValueError("No JSON object found")
            parsed, _end = decoder.raw_decode(content[start:])
            report = DraftReport.model_validate(parsed)
        except Exception as exc:
            print("Model JSON validation failed; finish_reason:", finish_reason)
            print("Model response prefix:", repr(content[:1200]))
            raise ValueError(
                f"Model returned invalid JSON or report fields; finish_reason={finish_reason}"
            ) from exc

    report.metadata = metadata
    report.review_required = True
    return report


@app.get("/health/live")
async def live():
    return {"status": "alive"}


@app.get("/health/ready", dependencies=[Depends(require_api_key)])
async def ready():
    return {"status": "ready" if llm is not None else "not_ready"}


@app.post("/v1/dicom/draft-report", response_model=DraftReport,
          dependencies=[Depends(require_api_key)])
async def create_draft_report(file: UploadFile = File(...)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds configured size limit")
    if not raw:
        raise HTTPException(400, "Empty upload")
    try:
        png, metadata = dicom_to_png(raw)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        return await run_in_threadpool(infer_sync, png, metadata)
    except ValueError as exc:
        raise HTTPException(502, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "Local inference failed") from exc