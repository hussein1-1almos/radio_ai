from __future__ import annotations
import asyncio, io, os, re, secrets, tempfile, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")  # before engine: it reads THREADS etc. at import
import engine  # noqa: E402  (sets thread limits before numpy/onnxruntime load)
import numpy as np  # noqa: E402
import pydicom  # noqa: E402
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402
from pydicom.pixels import apply_modality_lut, apply_voi_lut  # noqa: E402

API_KEY = os.getenv("API_KEY", "")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(64_000_000)))  # decoded rows*columns limit
MAX_EDGE = int(os.getenv("MAX_IMAGE_EDGE", "1024"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "4"))
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "300"))
IDLE_UNLOAD = float(os.getenv("IDLE_UNLOAD_SECONDS", "300"))
RETENTION = float(os.getenv("RETENTION_SECONDS", "3600"))  # annotated images are deleted after this
ANN_DIR = Path(os.getenv("ANNOTATED_DIR", Path(tempfile.gettempdir()) / "radai_annotated"))
INCLUDE_PHI = os.getenv("INCLUDE_PHI", "0") == "1"
ALLOWED_MODALITIES = {"CR", "DX", "DR", "CT", "MR", "MRI", "US"}  # CT/US are accepted and reported as unsupported
sem, waiting = asyncio.Semaphore(1), 0


class Annotation(BaseModel):
    id: str
    label: str
    box: list[float] = Field(description="[x_min, y_min, x_max, y_max], normalized 0-1 of the image analysed")
    box_format: str
    score: float | None = None
    score_calibrated: bool = False
    source_model: str
    model_version: str | None = None
    validated: bool = False
    image_ref: dict[str, Any] = Field(default_factory=dict)


class ModelRecord(BaseModel):
    name: str
    kind: str
    version: str | None = None
    validated: bool = False
    status: Literal["ran", "failed", "not_installed", "not_applicable"]
    detects: str = ""
    detail: str = ""


class Findings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str = ""
    limitations: list[str] = Field(default_factory=list)
    needs_additional_images: bool = True
    possible_urgent_finding: bool | None = None  # never asserted automatically
    bounding_box: list[list[float]] = Field(default_factory=list, description="legacy client: [ymin, xmin, ymax, xmax] 0-1000, localized findings only")
    flagged: list[str] = Field(default_factory=list)


class DraftReport(BaseModel):
    model_config = ConfigDict(extra="ignore")
    examination: str = ""
    technique: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    findings: Findings = Field(default_factory=Findings)
    impression: list[str] = Field(default_factory=list)
    uncertainty: str = "High"
    annotations: list[Annotation] = Field(default_factory=list)
    region_hints: list[dict[str, Any]] = Field(default_factory=list, description="context only; not lesion localization")
    annotation_status: Literal["localized", "none_detected", "localization_unavailable", "unsupported", "analysis_failed"]
    annotated_image_available: bool = False
    annotated_image_url: str | None = None
    models: list[ModelRecord] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    review_required: bool = True


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not API_KEY:
        raise HTTPException(503, "API authentication is not configured")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not supplied or not secrets.compare_digest(supplied, API_KEY):
        raise HTTPException(401, "Invalid authentication credentials")


def to_uint8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    ok = np.isfinite(a)
    if not ok.any():
        raise ValueError("Pixel data contains no finite values")
    lo, hi = np.percentile(a[ok], [1, 99])
    if hi <= lo:
        lo, hi = float(a[ok].min()), float(a[ok].max())
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    out = (np.clip(a, lo, hi) - lo) / (hi - lo) * 255
    out[~ok] = 0
    return out.astype(np.uint8)


def dicom_to_image(raw: bytes) -> tuple[Image.Image, np.ndarray | None, dict[str, Any]]:
    """-> (8-bit grayscale PIL image <= MAX_EDGE, raw float pixels for MR only, metadata). The upload is never modified."""
    try:
        ds = pydicom.dcmread(io.BytesIO(raw), force=False)
    except Exception as exc:
        raise ValueError("Upload is not a readable DICOM file") from exc
    mod = str(getattr(ds, "Modality", "")).upper().strip()
    if mod not in ALLOWED_MODALITIES:
        raise ValueError(f"Unsupported modality: {mod or 'missing'}")
    if "PixelData" not in ds:
        raise ValueError("DICOM contains no PixelData")
    if int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise ValueError("Multiframe DICOM is not supported; submit single-frame 2D instances")
    if int(getattr(ds, "Rows", 0)) * int(getattr(ds, "Columns", 0)) > MAX_PIXELS:
        raise ValueError("Image dimensions exceed the configured limit")
    try:
        px = ds.pixel_array
    except Exception as exc:
        raise ValueError("Cannot decode pixels; an additional DICOM codec may be required") from exc
    photo = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    arr = None
    if px.ndim == 3:
        if px.shape[-1] not in (3, 4) or photo == "PALETTE COLOR":
            raise ValueError("Unsupported color pixel layout")
        img = Image.fromarray((px if px.dtype == np.uint8 else to_uint8(px))[..., :3]).convert("L")
    elif px.ndim == 2:
        try:
            px = apply_modality_lut(px, ds)
        except Exception:
            px = px.astype(np.float32)
        if mod in ("MR", "MRI"):
            arr = np.asarray(px, np.float32)
        try:
            g = to_uint8(np.asarray(apply_voi_lut(px, ds), np.float32))
        except Exception:
            g = to_uint8(px)
        img = Image.fromarray(255 - g if photo == "MONOCHROME1" else g, "L")
    else:
        raise ValueError("Only 2D images are supported")
    img.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.BILINEAR)  # boxes are normalized, so resizing keeps them valid

    def g(tag, default="Unknown"):
        v = getattr(ds, tag, None)
        if v is None or v == "":
            return default
        if isinstance(v, float):
            return float(v)
        if isinstance(v, int):
            return int(v)
        if type(v).__name__ == "MultiValue":
            return [float(x) for x in v]
        return str(v).strip()

    meta = {"patient_name": g("PatientName"), "patient_id": g("PatientID"), "patient_birth_date": g("PatientBirthDate"),
            "patient_sex": g("PatientSex"), "patient_age": g("PatientAge"), "patient_weight": g("PatientWeight", 0.0),
            "study_date": g("StudyDate"), "study_time": g("StudyTime"), "study_description": g("StudyDescription", ""),
            "series_number": g("SeriesNumber", 0), "series_description": g("SeriesDescription", ""),
            "body_part": g("BodyPartExamined", ""), "modality": mod, "accession_number": g("AccessionNumber"),
            "study_instance_uid": g("StudyInstanceUID", None), "series_instance_uid": g("SeriesInstanceUID", None),
            "sop_instance_uid": g("SOPInstanceUID", None), "instance_number": g("InstanceNumber", 0),
            "manufacturer": g("Manufacturer"), "manufacturer_model_name": g("ManufacturerModelName"),
            "institution_name": g("InstitutionName"), "rows": int(getattr(ds, "Rows", img.height)),
            "columns": int(getattr(ds, "Columns", img.width)), "slice_thickness": g("SliceThickness", 0.0),
            "pixel_spacing": g("PixelSpacing", [0.0, 0.0]), "image_orientation_patient": g("ImageOrientationPatient", []),
            "image_position_patient": g("ImagePositionPatient", []), "magnetic_field_strength": g("MagneticFieldStrength", 0.0),
            "kvp": g("KVP", 0.0), "repetition_time": g("RepetitionTime", 0.0), "echo_time": g("EchoTime", 0.0),
            "scope": "single instance only"}
    if not INCLUDE_PHI:
        for k in ("patient_name", "patient_id", "patient_birth_date"):
            meta.pop(k)
    return img, arr, meta


def save_annotated(im: Image.Image) -> str:
    tok = secrets.token_urlsafe(16)
    p = ANN_DIR / f"{tok}.png"
    im.save(p, "PNG")
    os.chmod(p, 0o600)
    return tok


def sweep() -> None:
    for p in ANN_DIR.glob("*.png"):
        if time.time() - p.stat().st_mtime > RETENTION:
            p.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if hasattr(os, "nice"):
        os.nice(int(os.getenv("NICE", "10")))  # yield CPU to other processes on the server
    ANN_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)

    async def reaper():
        while True:
            await asyncio.sleep(30)
            try:
                sweep()
                engine.unload_idle(IDLE_UNLOAD)
            except Exception as exc:
                print("reaper:", repr(exc))
    task = asyncio.create_task(reaper())
    yield
    task.cancel()


app = FastAPI(title="Radiology draft-report API", version="0.5.0",
              description="CPU-only, lazy-loaded specialist detectors routed by body part. Preliminary AI output; radiologist review required.",
              lifespan=lifespan)


@app.get("/health/live")
async def live():
    return {"status": "alive"}


@app.get("/health/ready", dependencies=[Depends(require_api_key)])
async def ready():
    return {"status": "ready", "loaded_detectors": list(engine._cache), "waiting": waiting}


@app.get("/v1/coverage", dependencies=[Depends(require_api_key)])
async def coverage():
    """site -> detectors with installed / validated flags and what each detects. An empty list = no detector for that site."""
    return engine.coverage()


@app.get("/v1/annotated/{token}", dependencies=[Depends(require_api_key)])
async def annotated(token: str):
    p = ANN_DIR / f"{token}.png"
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,32}", token) or not p.is_file():
        raise HTTPException(404, "Annotated image not found or expired")
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/v1/dicom/draft-report", response_model=DraftReport, dependencies=[Depends(require_api_key)])
async def create_draft_report(file: UploadFile = File(...), site: str = Form("")):
    global waiting
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds configured size limit")
    if not raw:
        raise HTTPException(400, "Empty upload")
    try:
        img, arr, meta = await run_in_threadpool(dicom_to_image, raw)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    del raw
    if waiting >= MAX_QUEUE:
        raise HTTPException(503, "Server busy, retry shortly")

    def work():
        report = engine.build_report(engine.analyze(img, arr, meta, site), meta)
        try:
            text = engine.narrate(report)
        except Exception as exc:
            print("narrative skipped:", repr(exc))
            report["findings"]["limitations"].append("Narrative text generation was unavailable; standard template text is shown.")
            text = None
        if text:
            report["findings"]["summary"], report["impression"] = text[0], text[1]
            report["models"].append({"name": "report_text", "kind": "language_model", "version": text[2], "validated": False,
                                     "status": "ran", "detects": "None: rewrites the detector summary only; sees no pixels", "detail": ""})
        if report["annotations"]:
            report["annotated_image_url"] = f"/v1/annotated/{save_annotated(engine.annotate(img, report['annotations']))}"
            report["annotated_image_available"] = True
        return report

    waiting += 1
    try:
        async with sem:
            report = await asyncio.wait_for(run_in_threadpool(work), INFERENCE_TIMEOUT)
    except asyncio.TimeoutError as exc:  # the worker thread cannot be cancelled; it finishes in the background
        raise HTTPException(504, "Inference timed out") from exc
    except Exception as exc:
        print("inference error:", repr(exc))
        raise HTTPException(503, "Local inference failed") from exc
    finally:
        waiting -= 1
    report["metadata"] = meta
    report["review_required"] = True
    return report